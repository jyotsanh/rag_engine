"""DOCX extractor: python-docx → Markdown (headings, lists, tables), metadata, guards."""

import io
import zipfile
from collections.abc import AsyncIterator

import pytest
from docx import Document
from docx.document import Document as DocxDocument
from docx.enum.style import WD_STYLE_TYPE
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

from rag_engine.config import Settings
from rag_engine.crawling.fetchers import HttpFetcher
from rag_engine.domain import ContentKind, PageReason
from rag_engine.extraction import (
    DocumentTooLargeError,
    DocxExtractor,
    EmptyContentError,
    ExtractedDocument,
    ExtractionError,
    ExtractionInput,
    ExtractorRegistry,
)
from rag_engine.security import SafeHttpClient
from tests.conftest import OverrideSettings
from tests.support.fixture_site import SITE_ROOT, FixtureSite

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _bytes(doc: DocxDocument) -> bytes:
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


async def _extract(body: bytes, config: Settings | None = None) -> ExtractedDocument:
    extractor = DocxExtractor(config) if config is not None else DocxExtractor()
    source = ExtractionInput(url="https://acme.example/guide.docx", mime_type=_DOCX_MIME, body=body)
    return await extractor.extract(source)


def _add_hyperlink(paragraph: Paragraph, url: str, text: str) -> None:
    rel_id = paragraph.part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), rel_id)
    run = OxmlElement("w:r")
    text_element = OxmlElement("w:t")
    text_element.text = text
    run.append(text_element)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def _style_with_outline_level(doc: DocxDocument, name: str, level: int) -> None:
    style = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    outline = OxmlElement("w:outlineLvl")
    outline.set(qn("w:val"), str(level))
    style.element.get_or_add_pPr().append(outline)


@pytest.fixture
async def fetcher(override_settings: OverrideSettings) -> AsyncIterator[HttpFetcher]:
    config: Settings = override_settings(SSRF_ALLOW_PRIVATE_NETWORKS=True)
    async with SafeHttpClient(config) as client:
        yield HttpFetcher(client, config)


# ---- #
# Tests
# ---- #


class TestDocxExtractor_S42:
    """S42 — DOCX extractor: python-docx → Markdown (heading styles, lists, tables)."""

    async def test_fixture_install_guide(self) -> None:
        document = await _extract((SITE_ROOT / "files" / "install-guide.docx").read_bytes())

        assert document.markdown == (
            "# Installation Guide\n\n"
            "This guide covers mounting and wiring the WX-2048 widget.\n\n"
            "## Mounting\n\n"
            "- Use four M6 bolts torqued to 9 Nm.\n"
            "- Leave 50 mm clearance for airflow.\n\n"
            "## Wiring\n\n"
            "| Pin | Signal |\n|---|---|\n| 1 | +24 V |\n| 3 | GND |\n"
        )
        assert document.title == "Installation Guide"
        assert document.metadata == {
            "author": "Acme Widgets",
            "created": "2026-01-01",
            "modified": "2026-01-01",
        }
        assert (document.content_kind, document.extractor) == (ContentKind.DOCX, "docx")

    # ---- headings ---- #

    async def test_heading_styles(self) -> None:
        doc = Document()
        doc.add_paragraph("Release Notes", style="Title")
        for level in (1, 2, 3, 7):
            doc.add_heading(f"Level {level}", level=level)
        custom = doc.styles.add_style("Section Title", WD_STYLE_TYPE.PARAGRAPH)
        custom.base_style = doc.styles["Heading 2"]
        doc.add_paragraph("Custom heading", style="Section Title")

        markdown = (await _extract(_bytes(doc))).markdown

        assert markdown == (
            "# Release Notes\n\n"
            "# Level 1\n\n"
            "## Level 2\n\n"
            "### Level 3\n\n"
            "###### Level 7\n\n"  # capped at 6
            "## Custom heading\n"  # a style based on Heading 2
        )

    async def test_localized_heading_uses_the_outline_level(self) -> None:
        doc = Document()
        _style_with_outline_level(doc, "Überschrift 1", 0)
        _style_with_outline_level(doc, "Überschrift 2", 1)
        doc.add_paragraph("Einleitung", style="Überschrift 1")
        doc.add_paragraph("Montage", style="Überschrift 2")
        doc.add_paragraph("Vier Schrauben verwenden.")

        markdown = (await _extract(_bytes(doc))).markdown

        assert markdown.startswith("# Einleitung\n\n## Montage\n\nVier Schrauben verwenden.")

    # ---- lists & inline ---- #

    async def test_lists(self) -> None:
        doc = Document()
        doc.add_paragraph("Unload the belt.", style="List Number")
        doc.add_paragraph("Attach the weight.", style="List Number")
        doc.add_paragraph("Use the 5 kg weight.", style="List Bullet 2")
        doc.add_paragraph("Run the command.", style="List Number")
        doc.add_paragraph("Done.")

        markdown = (await _extract(_bytes(doc))).markdown

        assert markdown == (
            "1. Unload the belt.\n"
            "1. Attach the weight.\n"
            "   - Use the 5 kg weight.\n"
            "1. Run the command.\n\n"
            "Done.\n"
        )

    async def test_bold_italic_and_merged_runs(self) -> None:
        doc = Document()
        paragraph = doc.add_paragraph()
        for text, bold, italic in [
            ("Calibrate ", False, False),
            ("every ", True, False),
            ("quarter", True, False),  # merged with the previous bold run
            (" or after ", False, False),
            ("any", False, True),
            (" repair.", False, False),
        ]:
            run = paragraph.add_run(text)
            run.bold, run.italic = bold, italic

        markdown = (await _extract(_bytes(doc))).markdown

        assert markdown == "Calibrate **every quarter** or after *any* repair.\n"

    async def test_hyperlinks_keep_text_and_become_outlinks(self) -> None:
        doc = Document()
        paragraph = doc.add_paragraph("See the ")
        _add_hyperlink(paragraph, "https://acme.example/docs/", "online docs")
        paragraph.add_run(" for details.")
        _add_hyperlink(doc.add_paragraph(), "mailto:support@acme.example", "Email us")
        _add_hyperlink(doc.add_paragraph(), "https://acme.example/docs/", "again")

        document = await _extract(_bytes(doc))

        assert "See the online docs for details." in document.markdown
        assert document.outlinks == ("https://acme.example/docs/",)

    async def test_code_and_quote_styles(self) -> None:
        doc = Document()
        doc.styles.add_style("Source Code", WD_STYLE_TYPE.PARAGRAPH)
        doc.add_paragraph("Run these commands:")
        doc.add_paragraph("widgetctl update --all", style="Source Code")
        doc.add_paragraph("    widgetctl status", style="Source Code")
        doc.add_paragraph("Never skip the update.", style="Quote")

        markdown = (await _extract(_bytes(doc))).markdown

        assert markdown == (
            "Run these commands:\n\n"
            "```\nwidgetctl update --all\n    widgetctl status\n```\n\n"
            "> Never skip the update.\n"
        )

    # ---- tables ---- #

    async def test_tables_in_document_order(self) -> None:
        doc = Document()
        doc.add_heading("Limits", level=2)
        table = doc.add_table(rows=4, cols=3)
        values = [
            ("Model", "Torque | max", "Notes"),
            ("WX-2048", "12 Nm", "first line"),
            ("WX-4096", "18 Nm", ""),
            ("WX-8192", "", ""),
        ]
        for row, cells in zip(table.rows, values, strict=True):
            for cell, value in zip(row.cells, cells, strict=True):
                cell.text = value
        table.cell(1, 2).add_paragraph("second line")
        table.cell(2, 2).merge(table.cell(3, 2)).text = "shared note"  # vertical merge
        table.cell(3, 0).merge(table.cell(3, 1))  # horizontal merge
        doc.add_paragraph("After the table.")

        markdown = (await _extract(_bytes(doc))).markdown

        assert markdown == (
            "## Limits\n\n"
            "| Model | Torque \\| max | Notes |\n"
            "|---|---|---|\n"
            "| WX-2048 | 12 Nm | first line second line |\n"
            "| WX-4096 | 18 Nm | shared note |\n"
            "| WX-8192 |  |  |\n\n"
            "After the table.\n"
        )

    async def test_nested_table_is_flattened(self) -> None:
        doc = Document()
        outer = doc.add_table(rows=2, cols=2)
        outer.cell(0, 0).text, outer.cell(0, 1).text = "Part", "Spec"
        outer.cell(1, 0).text = "Sensor"
        inner = outer.cell(1, 1).add_table(rows=1, cols=2)
        inner.cell(0, 0).text, inner.cell(0, 1).text = "range", "0-50 Nm"

        markdown = (await _extract(_bytes(doc))).markdown

        assert "| Sensor | range 0-50 Nm |" in markdown

    # ---- metadata ---- #

    async def test_core_properties(self) -> None:
        doc = Document()
        doc.add_heading("Fallback title", level=1)
        doc.add_paragraph("Body.")
        doc.core_properties.title = ""
        doc.core_properties.subject = "Wiring the controller"
        doc.core_properties.language = "de-DE"

        document = await _extract(_bytes(doc))

        assert document.title == "Fallback title"  # first heading when the property is empty
        assert document.description == "Wiring the controller"
        assert document.lang == "de"

    async def test_language_falls_back_to_document_defaults(self) -> None:
        doc = Document()  # python-docx's template declares en-US run defaults
        doc.add_paragraph("Hello.")

        assert (await _extract(_bytes(doc))).lang == "en"

    # ---- failures & guards ---- #

    async def test_empty_document(self) -> None:
        doc = Document()
        doc.add_paragraph("   ")

        with pytest.raises(EmptyContentError):
            await _extract(_bytes(doc))

    @pytest.mark.parametrize(
        "body",
        [
            b"PK\x03\x04 not really a zip",
            None,  # a zip with Office markers but no real document part
        ],
        ids=["not-a-zip", "missing-parts"],
    )
    async def test_corrupt_docx(self, body: bytes | None) -> None:
        if body is None:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr("[Content_Types].xml", "<Types/>")
                archive.writestr("word/document.xml", "<broken")
            body = buffer.getvalue()

        with pytest.raises(ExtractionError) as exc:
            await _extract(body)

        assert exc.value.reason is PageReason.EXTRACTION_FAILED
        assert exc.value.detail == "corrupt"

    async def test_zip_bomb_is_refused_before_parsing(
        self, override_settings: OverrideSettings
    ) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", "<Types/>")
            archive.writestr("word/document.xml", b"<w:document>" + b" " * 5_000_000)
        config: Settings = override_settings(OOXML_MAX_UNCOMPRESSED_BYTES=1_000_000)

        with pytest.raises(DocumentTooLargeError) as exc:
            await _extract(buffer.getvalue(), config)

        assert len(buffer.getvalue()) < 50_000  # tiny on the wire
        assert (exc.value.reason, exc.value.detail) == (PageReason.TOO_LARGE, "uncompressed")

    # ---- integration ---- #

    async def test_dispatched_by_bytes_through_the_registry(self) -> None:
        body = (SITE_ROOT / "files" / "install-guide.docx").read_bytes()
        source = ExtractionInput(
            url="https://acme.example/download?id=7",
            mime_type="application/octet-stream",
            body=body,
        )

        document = await ExtractorRegistry([DocxExtractor()]).extract(source)

        assert document.extractor == "docx"

    async def test_fixture_docx_over_http(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        result = await fetcher.fetch(fixture_site.url("/files/install-guide.docx"))

        document = await DocxExtractor().extract(ExtractionInput.from_fetch(result))

        assert result.content_kind is ContentKind.DOCX
        assert "| 3 | GND |" in document.markdown
