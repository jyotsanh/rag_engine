"""Extractor framework: magic-byte sniffing, kind resolution, registry, error contract."""

import asyncio
import io
import threading
import zipfile
from typing import Any, cast

import pytest

from rag_engine.config import Settings
from rag_engine.crawling.fetchers import FetchResult
from rag_engine.domain import ContentKind, PageReason, PageStatus
from rag_engine.extraction import (
    BaseExtractor,
    DocumentUnreadableError,
    EmptyContentError,
    ExtractedDocument,
    ExtractionError,
    ExtractionInput,
    ExtractorRegistrationError,
    ExtractorRegistry,
    Sniffed,
    UnsupportedContentError,
    default_registry,
    looks_like_text,
    sniff,
)
from tests.conftest import OverrideSettings
from tests.support.fixture_site import SITE_ROOT

FILES = SITE_ROOT / "files"
_OLE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 504
_HTML = b"<!doctype html><html><head><title>t</title></head><body>hi</body></html>"
_PNG = (SITE_ROOT / "images" / "logo.png").read_bytes()


def _file(name: str) -> bytes:
    return (FILES / name).read_bytes()


def _zip(*names: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in names:
            archive.writestr(name, "x")
    return buffer.getvalue()


def _source(
    body: bytes,
    mime: str = "",
    url: str = "https://acme.example/x",
    *,
    encoding: str | None = None,
) -> ExtractionInput:
    return ExtractionInput(url=url, mime_type=mime, body=body, encoding=encoding)


class _Fake(BaseExtractor):
    """Echoes the kind it was registered for."""

    name = "fake"
    kind = ContentKind.HTML

    async def extract(self, source: ExtractionInput) -> ExtractedDocument:
        return ExtractedDocument(markdown="# ok", content_kind=self.kind, extractor=self.name)


def _fake(kind: ContentKind) -> BaseExtractor:
    cls = type(f"_Fake{kind}", (_Fake,), {"name": f"fake-{kind}", "kind": kind})
    return cls()


class _Raising(_Fake):
    name = "raising"

    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def extract(self, source: ExtractionInput) -> ExtractedDocument:
        raise self.error


class _Slow(_Fake):
    name = "slow"

    async def extract(self, source: ExtractionInput) -> ExtractedDocument:
        await asyncio.sleep(10)
        raise AssertionError("unreachable")


@pytest.fixture
def registry(override_settings: OverrideSettings) -> ExtractorRegistry:
    config: Settings = override_settings()
    return ExtractorRegistry([_fake(kind) for kind in ContentKind], config=config)


# ---- #
# Tests
# ---- #


class TestExtractorFramework_S38:
    """S38 — Extractor framework: BaseExtractor, registry (MIME + extension + magic bytes)."""

    # ---- sniffing ---- #

    @pytest.mark.parametrize(
        ("name", "kind"),
        [
            ("widget-spec.pdf", ContentKind.PDF),
            ("scanned.pdf", ContentKind.PDF),
            ("encrypted.pdf", ContentKind.PDF),  # PDF encryption keeps the PDF header
            ("install-guide.docx", ContentKind.DOCX),
            ("roadmap.pptx", ContentKind.PPTX),
            ("pricing.xlsx", ContentKind.XLSX),
        ],
    )
    def test_sniff_real_documents(self, name: str, kind: ContentKind) -> None:
        assert sniff(_file(name)) is kind

    @pytest.mark.parametrize(
        "body",
        [
            _HTML,
            b"\xef\xbb\xbf  \n<!DOCTYPE HTML PUBLIC '-//W3C//DTD HTML 4.01//EN'><p>x",
            b'<?xml version="1.0" encoding="utf-8"?>\n<html xmlns="http://www.w3.org/1999/xhtml">',
            b"<!-- generated -->\n<head><title>x</title></head>",
            b"<BODY>legacy page</BODY>",
        ],
        ids=["doctype", "bom-uppercase", "xhtml", "comment-head", "body-only"],
    )
    def test_sniff_html(self, body: bytes) -> None:
        assert sniff(body) is ContentKind.HTML

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            (b"junk before the header\r\n%PDF-1.7\n...", ContentKind.PDF),
            (_OLE, Sniffed.OLE),
            (_zip("readme.txt", "data/a.csv"), Sniffed.ZIP),
            (_zip("[Content_Types].xml", "unknown/part.xml"), Sniffed.ZIP),
            (_zip("[Content_Types].xml", "word/document.xml"), ContentKind.DOCX),
            (b"PK\x03\x04 truncated zip", Sniffed.ZIP),
            (_PNG, None),
            (b'{"json": true}', None),
            (b"", None),
        ],
        ids=[
            "pdf-junk-prefix",
            "ole",
            "plain-zip",
            "ooxml-unknown-part",
            "minimal-docx",
            "broken-zip",
            "png",
            "json",
            "empty",
        ],
    )
    def test_sniff_other_bodies(self, body: bytes, expected: object) -> None:
        assert sniff(body) == expected

    @pytest.mark.parametrize(
        ("body", "text"),
        [
            (b"plain words\nand lines\t\r\n", True),
            ("café latin-1 page".encode("latin-1"), True),
            ("日本語".encode("shift_jis"), True),
            (_PNG, False),
            (b"\x00\x01\x02binary", False),
            (b"", False),
        ],
    )
    def test_looks_like_text(self, body: bytes, text: bool) -> None:
        assert looks_like_text(body) is text

    # ---- kind resolution ---- #

    @pytest.mark.parametrize(
        ("body", "mime", "url", "kind"),
        [
            (_file("widget-spec.pdf"), "application/pdf", "/f.pdf", ContentKind.PDF),
            (
                _file("pricing.xlsx"),
                MIMES_XLSX := "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "/p.xlsx",
                ContentKind.XLSX,
            ),
            # generic type: bytes decide (the URL says nothing)
            (
                _file("install-guide.docx"),
                "application/octet-stream",
                "/download?id=7",
                ContentKind.DOCX,
            ),
            # mislabelled binaries: bytes beat the label
            (_file("widget-spec.pdf"), "text/html", "/spec", ContentKind.PDF),
            (_file("roadmap.pptx"), "application/zip", "/roadmap", ContentKind.PPTX),
            # a "PDF" that is really an HTML login page
            (_HTML, "application/pdf", "/report.pdf", ContentKind.HTML),
            # plain text labelled text/html still has words
            (b"Just text, no markup.", "text/html", "/notes", ContentKind.HTML),
            ("<p>café</p>".encode("latin-1"), "text/html", "/fr", ContentKind.HTML),
            (b"no markup at all", "application/octet-stream", "/page.html", ContentKind.HTML),
        ],
        ids=[
            "pdf",
            "xlsx",
            "octet-stream-docx",
            "pdf-as-html",
            "pptx-as-zip",
            "html-as-pdf",
            "text-as-html",
            "latin1-html",
            "octet-stream-html-extension",
        ],
    )
    def test_detect_kind(
        self, registry: ExtractorRegistry, body: bytes, mime: str, url: str, kind: ContentKind
    ) -> None:
        assert registry.detect_kind(_source(body, mime, "https://acme.example" + url)) is kind

    def test_utf16_html_is_trusted_by_its_detected_encoding(
        self, registry: ExtractorRegistry
    ) -> None:
        body = "<p>Text</p>".encode("utf-16")

        source = _source(body, "text/html", encoding="utf-16")

        assert registry.detect_kind(source) is ContentKind.HTML

    @pytest.mark.parametrize(
        ("body", "mime", "url", "error", "detail"),
        [
            (
                _OLE,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "/locked.docx",
                DocumentUnreadableError,
                "encrypted_or_legacy_office",
            ),
            (
                _OLE,
                "application/octet-stream",
                "/locked.xlsx",
                DocumentUnreadableError,
                "encrypted_or_legacy_office",
            ),
            (_OLE, "application/msword", "/old.doc", UnsupportedContentError, None),
            (b"not a pdf at all", "application/pdf", "/x.pdf", UnsupportedContentError, "not_pdf"),
            (_PNG, "image/png", "/logo.png", UnsupportedContentError, None),
            (_PNG, "text/html", "/image", UnsupportedContentError, None),
            (_zip("a.txt"), "application/zip", "/a.zip", UnsupportedContentError, None),
            (b'{"a": 1}', "application/json", "/api", UnsupportedContentError, None),
            (b"anything", "", "/no-type", UnsupportedContentError, None),
        ],
        ids=[
            "encrypted-docx",
            "encrypted-xlsx-by-extension",
            "legacy-doc",
            "fake-pdf",
            "image",
            "binary-as-html",
            "zip",
            "json",
            "no-content-type",
        ],
    )
    def test_detect_kind_errors(
        self,
        registry: ExtractorRegistry,
        body: bytes,
        mime: str,
        url: str,
        error: type[ExtractionError],
        detail: str | None,
    ) -> None:
        with pytest.raises(error) as exc:
            registry.detect_kind(_source(body, mime, "https://acme.example" + url))

        if detail is not None:
            assert exc.value.detail == detail

    # ---- registry ---- #

    async def test_resolve_and_extract(self, registry: ExtractorRegistry) -> None:
        source = _source(_file("roadmap.pptx"), "application/octet-stream")

        extractor = registry.resolve(source)
        document = await registry.extract(source)

        assert extractor.kind is ContentKind.PPTX
        assert (document.content_kind, document.extractor) == (ContentKind.PPTX, "fake-pptx")

    def test_kind_without_an_extractor_is_unsupported(
        self, override_settings: OverrideSettings
    ) -> None:
        registry = ExtractorRegistry([_fake(ContentKind.HTML)], config=override_settings())

        with pytest.raises(UnsupportedContentError) as exc:
            registry.resolve(_source(_file("widget-spec.pdf"), "application/pdf"))

        assert exc.value.detail == "no_extractor:pdf"
        assert registry.kinds == {ContentKind.HTML}

    def test_a_kind_registers_once(self, registry: ExtractorRegistry) -> None:
        with pytest.raises(ExtractorRegistrationError, match="already handled"):
            registry.register(_fake(ContentKind.PDF))

    def test_extractors_teach_extra_labels(self, override_settings: OverrideSettings) -> None:
        class _Markdownish(_Fake):
            name = "html-plus"
            kind = ContentKind.HTML
            mime_types = frozenset({"text/x-server-parsed-html"})
            extensions = frozenset({".shtml"})

        registry = ExtractorRegistry([_Markdownish()], config=override_settings())

        by_mime = _source(b"words", "text/x-server-parsed-html")
        by_extension = _source(b"words", "application/octet-stream", "https://a.example/p.shtml")
        untyped = _source(b"words", "", "https://acme.example/page.shtml")  # missing = generic
        wrong_type = _source(b"words", "text/plain", "https://acme.example/page.shtml")

        assert registry.detect_kind(by_mime) is ContentKind.HTML
        assert registry.detect_kind(by_extension) is ContentKind.HTML
        assert registry.detect_kind(untyped) is ContentKind.HTML
        with pytest.raises(UnsupportedContentError):  # an explicit other type beats the URL
            registry.detect_kind(wrong_type)

    @pytest.mark.parametrize(
        ("error", "reason", "status"),
        [
            (DocumentUnreadableError("locked"), PageReason.DOCUMENT_UNREADABLE, PageStatus.FAILED),
            (EmptyContentError("nothing"), PageReason.EMPTY_CONTENT, PageStatus.SKIPPED),
            (ExtractionError("bad"), PageReason.EXTRACTION_FAILED, PageStatus.FAILED),
        ],
    )
    async def test_typed_errors_pass_through(
        self,
        override_settings: OverrideSettings,
        error: ExtractionError,
        reason: PageReason,
        status: PageStatus,
    ) -> None:
        registry = ExtractorRegistry([_Raising(error)], config=override_settings())

        with pytest.raises(type(error)) as exc:
            await registry.extract(_source(_HTML, "text/html"))

        assert exc.value is error
        assert (exc.value.reason, exc.value.page_status) == (reason, status)

    async def test_parser_crash_becomes_extraction_error(
        self, override_settings: OverrideSettings
    ) -> None:
        registry = ExtractorRegistry(
            [_Raising(KeyError("missing table"))], config=override_settings()
        )

        with pytest.raises(ExtractionError) as exc:
            await registry.extract(_source(_HTML, "text/html"))

        assert type(exc.value) is ExtractionError
        assert exc.value.reason is PageReason.EXTRACTION_FAILED
        assert exc.value.detail.startswith("KeyError")
        assert isinstance(exc.value.__cause__, KeyError)

    async def test_extraction_is_time_capped(self, override_settings: OverrideSettings) -> None:
        registry = ExtractorRegistry(
            [_Slow()], config=override_settings(EXTRACTION_TIMEOUT_SECONDS=0.2)
        )

        with pytest.raises(ExtractionError) as exc:
            await registry.extract(_source(_HTML, "text/html"))

        assert exc.value.detail == "timeout"

    def test_unsupported_content_is_a_skip(self) -> None:
        error = UnsupportedContentError("nope")

        assert (error.reason, error.page_status) == (
            PageReason.UNSUPPORTED_CONTENT_TYPE,
            PageStatus.SKIPPED,
        )

    # ---- contract ---- #

    def test_extractor_must_declare_name_and_kind(self) -> None:
        with pytest.raises(TypeError, match="must declare `kind`"):

            class _NoKind(BaseExtractor):  # type: ignore[misc]
                name = "nokind"

                async def extract(self, source: ExtractionInput) -> ExtractedDocument:
                    raise NotImplementedError

        with pytest.raises(TypeError, match="extensions"):

            class _BadExt(_Fake):  # type: ignore[misc]
                extensions = frozenset({"PDF"})

    def test_extracted_document_normalization(self) -> None:
        document = ExtractedDocument(
            markdown="# T",
            content_kind=ContentKind.HTML,
            extractor="x",
            title="  Title  ",
            lang=" EN ",
            description="   ",
            metadata={"pages": 3, "author": "A", "draft": False, "score": 0.5, "none": None},
            outlinks=cast(Any, ["https://a.example/"]),  # a list is accepted and frozen
        )

        assert (document.title, document.lang, document.description) == ("Title", "en", None)
        assert document.outlinks == ("https://a.example/",)
        with pytest.raises(TypeError, match="scalar"):
            ExtractedDocument(
                markdown="",
                content_kind=ContentKind.PDF,
                extractor="x",
                metadata=cast(Any, {"a": [1]}),
            )

    def test_extraction_input_from_fetch_result(self) -> None:
        result = FetchResult(
            requested_url="https://a.example/r",
            final_url="https://a.example/final",
            redirect_chain=("https://a.example/r", "https://a.example/final"),
            status_code=200,
            mime_type="text/html",
            content_kind=ContentKind.HTML,
            body=_HTML,
            text=_HTML.decode(),
            encoding="utf-8",
            bytes_downloaded=len(_HTML),
            elapsed_ms=3,
            headers={"content-language": "de"},
        )

        source = ExtractionInput.from_fetch(result)

        assert source.url == "https://a.example/final"
        assert (source.mime_type, source.encoding) == ("text/html", "utf-8")
        assert source.text == _HTML.decode() and source.body == _HTML
        assert source.headers == {"content-language": "de"}

    async def test_run_sync_uses_a_worker_thread(self) -> None:
        loop_thread = threading.get_ident()

        worker_thread = await BaseExtractor.run_sync(threading.get_ident)

        assert worker_thread != loop_thread


class TestDefaultRegistry_S44:
    """S44 — with the last extractor in place, `default_registry()` covers every kind."""

    def test_every_content_kind_has_an_extractor(self) -> None:
        assert default_registry().kinds == frozenset(ContentKind)

    @pytest.mark.parametrize(
        ("path", "extractor"),
        [
            ("docs/getting-started.html", "html"),
            ("files/widget-spec.pdf", "pdf"),
            ("files/install-guide.docx", "docx"),
            ("files/roadmap.pptx", "pptx"),
            ("files/pricing.xlsx", "xlsx"),
        ],
    )
    async def test_fixture_documents_dispatch_by_bytes(self, path: str, extractor: str) -> None:
        body = (SITE_ROOT / path).read_bytes()
        source = _source(body, "application/octet-stream", f"https://acme.example/{path}")

        document = await default_registry().extract(source)

        assert document.extractor == extractor
        assert document.markdown.strip()
