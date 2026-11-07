"""PDF extractor: pymupdf4llm Markdown with page markers, caps, unreadable PDFs."""

import asyncio
from collections.abc import AsyncIterator

import pymupdf
import pytest

import rag_engine.extraction.pdf as pdf_module
from rag_engine.config import Settings
from rag_engine.crawling.fetchers import HttpFetcher
from rag_engine.domain import ContentKind, PageReason, PageStatus
from rag_engine.extraction import (
    DocumentTooLargeError,
    DocumentUnreadableError,
    ExtractedDocument,
    ExtractionError,
    ExtractionInput,
    ExtractorRegistry,
    PdfExtractor,
    page_marker,
)
from rag_engine.security import SafeHttpClient
from tests.conftest import OverrideSettings
from tests.support.fixture_site import SITE_ROOT, FixtureSite

FILES = SITE_ROOT / "files"


def _source(body: bytes, mime: str = "application/pdf") -> ExtractionInput:
    return ExtractionInput(url="https://acme.example/doc.pdf", mime_type=mime, body=body)


async def _extract(body: bytes, config: Settings | None = None) -> ExtractedDocument:
    extractor = PdfExtractor(config) if config is not None else PdfExtractor()
    return await extractor.extract(_source(body))


def _chapters(pages: int, *, header: bool = False, **save: object) -> bytes:
    doc = pymupdf.open()
    for n in range(1, pages + 1):
        page = doc.new_page()
        if header:  # running header and footer on every page
            page.insert_text((72, 40), "ACME CONFIDENTIAL - Widget Manual", fontsize=8)
            page.insert_text((280, 820), f"Page {n} of {pages}", fontsize=8)
        page.insert_text((72, 90), f"Chapter {n}", fontsize=18)
        page.insert_text(
            (72, 120), f"Chapter {n} explains the widget calibration procedure.", fontsize=11
        )
    return doc.tobytes(**save)  # type: ignore[arg-type]


def _table_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 90), "Torque limits", fontsize=18)
    rows = [
        ("Model", "Torque", "Speed"),
        ("WX-2048", "12 Nm", "1200/min"),
        ("WX-4096", "18 Nm", "2400/min"),
    ]
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(72 + c * 120, 120 + r * 24, 192 + c * 120, 144 + r * 24)
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            page.insert_text((rect.x0 + 6, rect.y0 + 16), cell, fontsize=10)
    page.insert_text(
        (72, 230), "Limits apply at 25 degrees Celsius ambient temperature.", fontsize=11
    )
    return doc.tobytes()


@pytest.fixture
async def fetcher(override_settings: OverrideSettings) -> AsyncIterator[HttpFetcher]:
    config: Settings = override_settings(SSRF_ALLOW_PRIVATE_NETWORKS=True)
    async with SafeHttpClient(config) as client:
        yield HttpFetcher(client, config)


# ---- #
# Tests
# ---- #


class TestPdfExtractor_S41:
    """S41 — PDF extractor: pymupdf4llm → Markdown (page markers, tables), caps, unreadable."""

    # ---- content ---- #

    async def test_fixture_spec_sheet(self) -> None:
        document = await _extract((FILES / "widget-spec.pdf").read_bytes())

        assert document.markdown.startswith(f"{page_marker(1)}\n\n# Widget Specification\n")
        assert f"{page_marker(2)}\n\n# Electrical Interface" in document.markdown
        assert "RS-485 data" in document.markdown
        assert document.title == "Widget Specification"
        assert document.metadata["author"] == "Acme Widgets"
        assert (document.metadata["page_count"], document.metadata["pages_extracted"]) == (2, 2)
        assert document.metadata["truncated"] is False
        assert document.metadata["extraction"] == "pymupdf4llm"
        assert (document.content_kind, document.extractor) == (ContentKind.PDF, "pdf")

    async def test_tables_become_markdown_tables(self) -> None:
        markdown = (await _extract(_table_pdf())).markdown

        assert "|Model|Torque|Speed|" in markdown.replace(" ", "")
        assert "|WX-4096|18Nm|2400/min|" in markdown.replace(" ", "")
        assert "Limits apply at 25 degrees Celsius" in markdown

    async def test_running_headers_and_footers_are_dropped(self) -> None:
        markdown = (await _extract(_chapters(3, header=True))).markdown

        assert "ACME CONFIDENTIAL" not in markdown
        assert "of 3" not in markdown
        assert [f"# Chapter {n}" in markdown for n in (1, 2, 3)] == [True] * 3

    async def test_page_markers_are_in_order(self) -> None:
        markdown = (await _extract(_chapters(4))).markdown

        positions = [markdown.index(page_marker(n)) for n in (1, 2, 3, 4)]
        assert positions == sorted(positions)

    # ---- metadata ---- #

    async def test_metadata_from_the_document(self) -> None:
        doc = pymupdf.open(stream=_chapters(1), filetype="pdf")
        doc.set_metadata(
            {"title": "Calibration Manual", "subject": "How to calibrate widgets",
             "creationDate": "D:20250314093000+01'00'"}
        )  # fmt: skip
        doc.xref_set_key(doc.pdf_catalog(), "Lang", "(de-DE)")

        document = await _extract(doc.tobytes())

        assert document.title == "Calibration Manual"
        assert document.description == "How to calibrate widgets"
        assert document.lang == "de"
        assert document.metadata["created"] == "2025-03-14"

    @pytest.mark.parametrize("junk", ["", "Untitled", "Microsoft Word - draft_v3.docx", "scan.pdf"])
    async def test_title_falls_back_to_the_first_heading(self, junk: str) -> None:
        doc = pymupdf.open(stream=_chapters(2), filetype="pdf")
        doc.set_metadata({"title": junk})

        assert (await _extract(doc.tobytes())).title == "Chapter 1"

    async def test_uri_links_become_outlinks(self) -> None:
        doc = pymupdf.open(stream=_chapters(2), filetype="pdf")
        for uri in (
            "https://acme.example/docs/",
            "mailto:a@acme.example",
            "https://acme.example/docs/",
        ):
            doc[0].insert_link(
                {"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(72, 110, 300, 125), "uri": uri}
            )
        doc[1].insert_link(
            {
                "kind": pymupdf.LINK_URI,
                "from": pymupdf.Rect(72, 110, 300, 125),
                "uri": "http://other.example/x",
            }
        )

        document = await _extract(doc.tobytes())

        assert document.outlinks == ("https://acme.example/docs/", "http://other.example/x")

    # ---- caps ---- #

    async def test_page_cap_truncates(self, override_settings: OverrideSettings) -> None:
        config: Settings = override_settings(PDF_MAX_PAGES=2)

        document = await _extract(_chapters(5), config)

        assert "# Chapter 2" in document.markdown and "Chapter 3" not in document.markdown
        assert document.metadata["truncated"] is True
        assert (document.metadata["page_count"], document.metadata["pages_extracted"]) == (5, 2)

    async def test_byte_cap(self, override_settings: OverrideSettings) -> None:
        config: Settings = override_settings(PDF_MAX_BYTES=1000)

        with pytest.raises(DocumentTooLargeError) as exc:
            await _extract((FILES / "widget-spec.pdf").read_bytes(), config)

        assert (exc.value.reason, exc.value.page_status) == (
            PageReason.TOO_LARGE,
            PageStatus.FAILED,
        )

    # ---- unreadable & broken ---- #

    async def test_scanned_pdf_has_no_text_layer(self) -> None:
        with pytest.raises(DocumentUnreadableError) as exc:
            await _extract((FILES / "scanned.pdf").read_bytes())

        assert exc.value.detail == "no_text_layer"
        assert (exc.value.reason, exc.value.page_status) == (
            PageReason.DOCUMENT_UNREADABLE,
            PageStatus.FAILED,
        )

    async def test_password_protected_pdf(self) -> None:
        with pytest.raises(DocumentUnreadableError) as exc:
            await _extract((FILES / "encrypted.pdf").read_bytes())

        assert exc.value.detail == "encrypted"

    async def test_owner_password_only_pdf_is_readable(self) -> None:
        body = _chapters(
            1,
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner-only",
            permissions=pymupdf.PDF_PERM_PRINT,
        )

        document = await _extract(body)

        assert "# Chapter 1" in document.markdown

    @pytest.mark.parametrize(
        "body", [b"%PDF-1.7\nthis is not really a pdf", b"%PDF-1.4\n" + b"\x00" * 64]
    )
    async def test_corrupt_pdf_fails_extraction(self, body: bytes) -> None:
        with pytest.raises(ExtractionError) as exc:
            await _extract(body)

        assert exc.value.reason is PageReason.EXTRACTION_FAILED

    async def test_layout_engine_failure_falls_back_to_the_text_layer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(*_args: object, **_kwargs: object) -> list:
            raise RuntimeError("layout model crashed")

        monkeypatch.setattr(pdf_module.pymupdf4llm, "to_markdown", broken)

        document = await _extract(_chapters(2))

        assert document.metadata["extraction"] == "pymupdf_text"
        assert f"{page_marker(2)}\n\nChapter 2" in document.markdown

    # ---- integration ---- #

    async def test_concurrent_extractions_are_serialized_safely(self) -> None:
        bodies = [_chapters(n) for n in (1, 2, 3, 4)]

        documents = await asyncio.gather(*(_extract(body) for body in bodies))

        assert [d.metadata["page_count"] for d in documents] == [1, 2, 3, 4]

    async def test_dispatched_by_bytes_through_the_registry(self) -> None:
        registry = ExtractorRegistry([PdfExtractor()])
        source = _source((FILES / "widget-spec.pdf").read_bytes(), "application/octet-stream")

        document = await registry.extract(source)

        assert document.extractor == "pdf"

    async def test_fixture_pdf_over_http(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        result = await fetcher.fetch(fixture_site.url("/files/widget-spec.pdf"))

        document = await PdfExtractor().extract(ExtractionInput.from_fetch(result))

        assert result.content_kind is ContentKind.PDF
        assert "# Widget Specification" in document.markdown
