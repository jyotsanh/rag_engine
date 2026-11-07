"""PPTX extractor: a section per slide (title, body, tables, notes), metadata, caps."""

import io
import zipfile
from collections.abc import AsyncIterator, Callable

import pytest
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.presentation import Presentation as PresentationType
from pptx.util import Inches

from rag_engine.config import Settings
from rag_engine.crawling.fetchers import HttpFetcher
from rag_engine.domain import ContentKind, PageReason
from rag_engine.extraction import (
    DocumentTooLargeError,
    EmptyContentError,
    ExtractedDocument,
    ExtractionError,
    ExtractionInput,
    ExtractorRegistry,
    PptxExtractor,
    slide_marker,
)
from rag_engine.security import SafeHttpClient
from tests.conftest import OverrideSettings
from tests.support.fixture_site import SITE_ROOT, FixtureSite

_PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
_TITLE_AND_CONTENT, _TITLE_ONLY, _BLANK = 1, 5, 6

type Build = Callable[[PresentationType], None]


def _deck(*builders: Build) -> bytes:
    prs = Presentation()
    for build in builders:
        build(prs)
    buffer = io.BytesIO()
    prs.save(buffer)
    return buffer.getvalue()


def _bullets(title: str, *items: tuple[str, int], notes: str | None = None) -> Build:
    def build(prs: PresentationType) -> None:
        slide = prs.slides.add_slide(prs.slide_layouts[_TITLE_AND_CONTENT])
        slide.shapes.title.text = title
        frame = slide.placeholders[1].text_frame
        for i, (text, level) in enumerate(items):
            paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            paragraph.text, paragraph.level = text, level
        if notes is not None:
            slide.notes_slide.notes_text_frame.text = notes

    return build


async def _extract(body: bytes, config: Settings | None = None) -> ExtractedDocument:
    extractor = PptxExtractor(config) if config is not None else PptxExtractor()
    source = ExtractionInput(url="https://acme.example/deck.pptx", mime_type=_PPTX_MIME, body=body)
    return await extractor.extract(source)


@pytest.fixture
async def fetcher(override_settings: OverrideSettings) -> AsyncIterator[HttpFetcher]:
    config: Settings = override_settings(SSRF_ALLOW_PRIVATE_NETWORKS=True)
    async with SafeHttpClient(config) as client:
        yield HttpFetcher(client, config)


# ---- #
# Tests
# ---- #


class TestPptxExtractor_S43:
    """S43 — PPTX extractor: a section per slide (title, body, tables, speaker notes)."""

    async def test_fixture_roadmap(self) -> None:
        document = await _extract((SITE_ROOT / "files" / "roadmap.pptx").read_bytes())

        assert document.markdown == (
            f"{slide_marker(1)}\n\n## Product Roadmap 2026\n\n"
            "- Widget Controller v3 ships in Q2 with fleet-wide OTA updates.\n\n"
            "> Speaker notes: emphasise product roadmap 2026.\n\n"
            f"{slide_marker(2)}\n\n## Q3: Open Widget Protocol 2.0\n\n"
            "- Adds encrypted telemetry and frame compression.\n\n"
            "> Speaker notes: emphasise q3: open widget protocol 2.0.\n\n"
            f"{slide_marker(3)}\n\n## Release dates\n\n"
            "| Release | Date |\n|---|---|\n| v3.0 | 2026-05-01 |\n| v3.1 | 2026-08-15 |\n"
        )
        assert document.title == "Product Roadmap 2026"
        assert document.metadata["slide_count"] == 3
        assert document.metadata["created"] == "2026-01-01"
        assert (document.content_kind, document.extractor) == (ContentKind.PPTX, "pptx")

    # ---- slide content ---- #

    async def test_nested_bullets_and_multi_paragraph_notes(self) -> None:
        body = _deck(
            _bullets(
                "Calibration",
                ("Unload the belt", 0),
                ("Check the rollers", 1),
                ("Clean the sensor\vwith a dry cloth", 2),  # soft line break
                ("Attach the weight", 0),
                notes="Mention the 5 kg kit.\nAsk about recent repairs.",
            )
        )

        markdown = (await _extract(body)).markdown

        assert markdown == (
            f"{slide_marker(1)}\n\n## Calibration\n\n"
            "- Unload the belt\n"
            "   - Check the rollers\n"
            "      - Clean the sensor with a dry cloth\n"
            "- Attach the weight\n\n"
            "> Mention the 5 kg kit.\n> Ask about recent repairs.\n"
        )

    async def test_text_boxes_in_reading_order(self) -> None:
        def build(prs: PresentationType) -> None:
            slide = prs.slides.add_slide(prs.slide_layouts[_TITLE_ONLY])
            slide.shapes.title.text = "Layout"
            # inserted bottom-right first; read top-to-bottom, left-to-right
            for left, top, text in [
                (5, 5, "Bottom right"),
                (1, 5, "Bottom left"),
                (1, 3, "Middle"),
            ]:
                slide.shapes.add_textbox(
                    Inches(left), Inches(top), Inches(3), Inches(1)
                ).text = text

        markdown = (await _extract(_deck(build))).markdown

        assert markdown.endswith("## Layout\n\nMiddle\n\nBottom left\n\nBottom right\n")

    async def test_tables_with_merged_cells(self) -> None:
        def build(prs: PresentationType) -> None:
            slide = prs.slides.add_slide(prs.slide_layouts[_TITLE_ONLY])
            slide.shapes.title.text = "Limits"
            table = slide.shapes.add_table(3, 3, Inches(1), Inches(2), Inches(6), Inches(2)).table
            values = [
                ("Model", "Torque | max", "Speed"),
                ("WX-2048", "12 Nm", "1200"),
                ("All", "", ""),
            ]
            for r, row in enumerate(values):
                for c, value in enumerate(row):
                    table.cell(r, c).text = value
            table.cell(2, 0).merge(table.cell(2, 2))

        markdown = (await _extract(_deck(build))).markdown

        assert (
            "| Model | Torque \\| max | Speed |\n|---|---|---|\n"
            "| WX-2048 | 12 Nm | 1200 |\n| All |  |  |"
        ) in markdown

    async def test_group_shapes_and_chart_titles(self) -> None:
        def build(prs: PresentationType) -> None:
            slide = prs.slides.add_slide(prs.slide_layouts[_TITLE_ONLY])
            slide.shapes.title.text = "Fleet"
            group = slide.shapes.add_group_shape()
            group.shapes.add_textbox(
                Inches(1), Inches(2), Inches(3), Inches(1)
            ).text = "Grouped note"
            data = CategoryChartData()
            data.categories = ["Q1", "Q2"]
            data.add_series("Units", (120, 180))
            chart = slide.shapes.add_chart(
                XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(4), Inches(6), Inches(3), data
            ).chart
            chart.has_title = True
            chart.chart_title.text_frame.text = "Units shipped per quarter"

        markdown = (await _extract(_deck(build))).markdown

        assert "Grouped note" in markdown
        assert "Chart: Units shipped per quarter" in markdown

    async def test_untitled_slide_and_empty_slide(self) -> None:
        def untitled(prs: PresentationType) -> None:
            slide = prs.slides.add_slide(prs.slide_layouts[_BLANK])
            slide.shapes.add_textbox(
                Inches(1), Inches(1), Inches(4), Inches(1)
            ).text = "Just a note"

        def empty(prs: PresentationType) -> None:
            prs.slides.add_slide(prs.slide_layouts[_BLANK])

        markdown = (await _extract(_deck(_bullets("Intro", ("Hi", 0)), empty, untitled))).markdown

        assert f"{slide_marker(3)}\n\n## Slide 3\n\nJust a note" in markdown
        assert slide_marker(2) not in markdown  # nothing on it

    # ---- visibility & caps ---- #

    async def test_hidden_slides_are_skipped(self) -> None:
        def hidden(prs: PresentationType) -> None:
            _bullets("Backup", ("Old numbers", 0))(prs)
            prs.slides[-1]._element.set("show", "0")

        document = await _extract(_deck(_bullets("Visible", ("Kept", 0)), hidden))

        assert "Backup" not in document.markdown
        assert document.metadata["hidden_skipped"] == 1
        assert document.metadata["slides_extracted"] == 1

    async def test_slide_cap_truncates(self, override_settings: OverrideSettings) -> None:
        body = _deck(*(_bullets(f"Slide title {n}", (f"Point {n}", 0)) for n in range(1, 6)))

        document = await _extract(body, override_settings(PPTX_MAX_SLIDES=2))

        assert "Slide title 2" in document.markdown and "Slide title 3" not in document.markdown
        assert document.metadata["truncated"] is True
        assert (document.metadata["slide_count"], document.metadata["slides_extracted"]) == (5, 2)

    # ---- metadata & links ---- #

    async def test_core_properties_and_language(self) -> None:
        def props(prs: PresentationType) -> None:
            _bullets("Roadmap", ("Q2 launch", 0))(prs)
            core = prs.core_properties
            core.title, core.subject, core.author, core.language = (
                "Board deck",
                "2026 plan",
                "Dana",
                "fr-FR",
            )

        document = await _extract(_deck(props))

        assert (document.title, document.description, document.lang) == (
            "Board deck",
            "2026 plan",
            "fr",
        )
        assert document.metadata["author"] == "Dana"

    async def test_language_from_text_runs(self) -> None:
        def spanish(prs: PresentationType) -> None:
            _bullets("Hoja de ruta", ("Lanzamiento", 0))(prs)
            run = prs.slides[0].placeholders[1].text_frame.paragraphs[0].runs[0]
            run._r.get_or_add_rPr().set("lang", "es-ES")

        assert (await _extract(_deck(spanish))).lang == "es"

    async def test_hyperlinks_become_outlinks(self) -> None:
        def links(prs: PresentationType) -> None:
            slide = prs.slides.add_slide(prs.slide_layouts[_TITLE_ONLY])
            slide.shapes.title.text = "Links"
            box = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(4), Inches(1))
            run = box.text_frame.paragraphs[0].add_run()
            run.text = "Docs"
            run.hyperlink.address = "https://acme.example/docs/"
            mail = box.text_frame.add_paragraph().add_run()
            mail.text, mail.hyperlink.address = "Mail", "mailto:a@acme.example"
            button = slide.shapes.add_textbox(Inches(1), Inches(4), Inches(2), Inches(1))
            button.text = "Shop"
            button.click_action.hyperlink.address = "https://shop.acme.example/"

        document = await _extract(_deck(links))

        assert document.outlinks == ("https://acme.example/docs/", "https://shop.acme.example/")

    # ---- failures ---- #

    async def test_deck_without_text(self) -> None:
        def empty(prs: PresentationType) -> None:
            prs.slides.add_slide(prs.slide_layouts[_BLANK])

        with pytest.raises(EmptyContentError):
            await _extract(_deck(empty))

    async def test_corrupt_pptx(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("[Content_Types].xml", "<Types/>")
            archive.writestr("ppt/presentation.xml", "<broken")

        with pytest.raises(ExtractionError) as exc:
            await _extract(buffer.getvalue())

        assert (exc.value.reason, exc.value.detail) == (PageReason.EXTRACTION_FAILED, "corrupt")

    async def test_zip_bomb_guard_applies(self, override_settings: OverrideSettings) -> None:
        body = _deck(_bullets("Big", ("x" * 2000, 0)))

        with pytest.raises(DocumentTooLargeError):
            await _extract(body, override_settings(OOXML_MAX_UNCOMPRESSED_BYTES=1000))

    # ---- integration ---- #

    async def test_dispatched_by_bytes_through_the_registry(self) -> None:
        body = (SITE_ROOT / "files" / "roadmap.pptx").read_bytes()
        source = ExtractionInput(
            url="https://acme.example/f", mime_type="application/zip", body=body
        )

        document = await ExtractorRegistry([PptxExtractor()]).extract(source)

        assert document.extractor == "pptx"

    async def test_fixture_pptx_over_http(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        result = await fetcher.fetch(fixture_site.url("/files/roadmap.pptx"))

        document = await PptxExtractor().extract(ExtractionInput.from_fetch(result))

        assert result.content_kind is ContentKind.PPTX
        assert "## Release dates" in document.markdown
