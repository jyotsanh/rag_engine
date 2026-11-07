"""XLSX extractor: a Markdown table per visible sheet, value formatting, caps."""

import io
import zipfile
from collections.abc import AsyncIterator
from datetime import date, datetime, time

import pytest
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference

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
    XlsxExtractor,
    sheet_marker,
)
from rag_engine.security import SafeHttpClient
from tests.conftest import OverrideSettings
from tests.support.fixture_site import SITE_ROOT, FixtureSite

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _bytes(workbook: Workbook) -> bytes:
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _book(*sheets: tuple[str, list[tuple]]) -> Workbook:
    workbook = Workbook()
    assert workbook.active is not None
    workbook.remove(workbook.active)
    for title, rows in sheets:
        sheet = workbook.create_sheet(title)
        for row in rows:
            sheet.append(row)
    return workbook


async def _extract(body: bytes, config: Settings | None = None) -> ExtractedDocument:
    extractor = XlsxExtractor(config) if config is not None else XlsxExtractor()
    source = ExtractionInput(url="https://acme.example/book.xlsx", mime_type=_XLSX_MIME, body=body)
    return await extractor.extract(source)


@pytest.fixture
async def fetcher(override_settings: OverrideSettings) -> AsyncIterator[HttpFetcher]:
    config: Settings = override_settings(SSRF_ALLOW_PRIVATE_NETWORKS=True)
    async with SafeHttpClient(config) as client:
        yield HttpFetcher(client, config)


# ---- #
# Tests
# ---- #


class TestXlsxExtractor_S44:
    """S44 — XLSX extractor: openpyxl read-only → a Markdown table per sheet, row/col caps."""

    async def test_fixture_price_list(self) -> None:
        document = await _extract((SITE_ROOT / "files" / "pricing.xlsx").read_bytes())

        assert document.markdown == (
            f"{sheet_marker(1)}\n\n## Prices\n\n"
            "| SKU | Description | Unit price (USD) |\n|---|---|---|\n"
            "| WX-2048 | Modular sorter | 1899 |\n"
            "| WX-4096 | High-throughput sorter | 3499 |\n"
            "| SNS-T1 | Torque sensor pack | 249 |\n\n"
            f"{sheet_marker(2)}\n\n## Discounts\n\n"
            "| Quantity | Discount |\n|---|---|\n| 10+ | 5% |\n| 50+ | 12% |\n"
        )
        assert document.metadata["sheet_count"] == 2
        assert document.metadata["created"] == "2026-01-01"
        assert (document.content_kind, document.extractor) == (ContentKind.XLSX, "xlsx")

    # ---- values ---- #

    async def test_value_formatting(self) -> None:
        workbook = _book(
            (
                "Values",
                [
                    ("Kind", "Value"),
                    ("int", 42),
                    ("integral float", 1899.0),
                    ("float", 2.5),
                    ("long float", 1 / 3),
                    ("date", date(2026, 5, 1)),
                    ("midnight", datetime(2026, 5, 1, 0, 0)),
                    ("datetime", datetime(2026, 5, 1, 14, 30)),
                    ("time", time(9, 15)),
                    ("bool", True),
                    ("text", "  multi\nline   text "),
                    ("pipe", "A | B"),
                    ("percent", 0.125),
                ],
            )
        )
        sheet = workbook["Values"]
        sheet["B13"].number_format = "0.0%"  # row 13: the "percent" row

        markdown = (await _extract(_bytes(workbook))).markdown

        for expected in [
            "| int | 42 |",
            "| integral float | 1899 |",
            "| float | 2.5 |",
            "| long float | 0.3333333333 |",
            "| date | 2026-05-01 |",
            "| midnight | 2026-05-01 |",
            "| datetime | 2026-05-01 14:30:00 |",
            "| time | 09:15:00 |",
            "| bool | TRUE |",
            "| text | multi line text |",
            "| pipe | A \\| B |",
            "| percent | 12.5% |",
        ]:
            assert expected in markdown, expected

    async def test_formulas_show_cached_values_only(self) -> None:
        # openpyxl writes formulas without cached results, as an unrecalculated file would
        workbook = _book(("Sums", [("a", "b", "total"), (1, 2, "=A2+B2")]))

        markdown = (await _extract(_bytes(workbook))).markdown

        assert "| 1 | 2 |  |" in markdown
        assert "=A2+B2" not in markdown

    # ---- shape of the table ---- #

    async def test_empty_rows_and_columns_are_trimmed(self) -> None:
        workbook = _book(("Sparse", []))
        sheet = workbook["Sparse"]
        sheet["C5"], sheet["E5"] = "Model", "Torque"  # data starts at C5; column D is empty
        sheet["C6"], sheet["E6"] = "WX-2048", "12 Nm"
        sheet["C9"], sheet["E9"] = "WX-4096", "18 Nm"  # rows 7-8 empty
        sheet["Z100"].number_format = "0.00"  # formatting far past the data, no value

        markdown = (await _extract(_bytes(workbook))).markdown

        assert markdown.endswith(
            "| Model | Torque |\n|---|---|\n| WX-2048 | 12 Nm |\n| WX-4096 | 18 Nm |\n"
        )

    async def test_title_row_becomes_a_caption(self) -> None:
        workbook = _book(("Prices", [("Price list 2026",), ("SKU", "Price"), ("WX-2048", 1899)]))

        markdown = (await _extract(_bytes(workbook))).markdown

        assert markdown == (
            f"{sheet_marker(1)}\n\n## Prices\n\nPrice list 2026\n\n"
            "| SKU | Price |\n|---|---|\n| WX-2048 | 1899 |\n"
        )

    async def test_single_column_sheet_keeps_its_header(self) -> None:
        workbook = _book(("Names", [("Name",), ("Alpha",), ("Beta",)]))

        markdown = (await _extract(_bytes(workbook))).markdown

        assert "| Name |\n|---|\n| Alpha |\n| Beta |" in markdown

    # ---- sheets ---- #

    async def test_hidden_empty_and_chart_sheets(self) -> None:
        workbook = _book(
            ("Visible", [("k", "v"), ("a", 1)]),
            ("Hidden", [("secret", 1)]),
            ("VeryHidden", [("secret", 2)]),
            ("Empty", []),
        )
        workbook["Hidden"].sheet_state = "hidden"
        workbook["VeryHidden"].sheet_state = "veryHidden"
        chart = BarChart()
        chart.add_data(Reference(workbook["Visible"], min_col=2, min_row=1, max_row=2))
        workbook.create_chartsheet("Chart").add_chart(chart)

        document = await _extract(_bytes(workbook))

        assert "## Visible" in document.markdown
        assert "secret" not in document.markdown
        assert "## Empty" not in document.markdown
        assert document.metadata["hidden_skipped"] == 2

    async def test_workbook_without_values(self) -> None:
        with pytest.raises(EmptyContentError):
            await _extract(_bytes(_book(("Blank", []), ("AlsoBlank", [(None, None)]))))

    # ---- caps ---- #

    async def test_row_cap(self, override_settings: OverrideSettings) -> None:
        rows = [("n", "square"), *((n, n * n) for n in range(1, 10))]

        document = await _extract(
            _bytes(_book(("Squares", rows))), override_settings(XLSX_MAX_ROWS=4)
        )

        assert "| 3 | 9 |" in document.markdown and "| 4 | 16 |" not in document.markdown
        assert "*Truncated: showing the first 4 rows and 50 columns.*" in document.markdown
        assert (document.metadata["truncated_sheets"], document.metadata["truncated"]) == (1, True)

    async def test_exactly_at_the_row_cap_is_not_truncated(
        self, override_settings: OverrideSettings
    ) -> None:
        rows = [("n",), (1,), (2,), (3,)]

        document = await _extract(_bytes(_book(("Four", rows))), override_settings(XLSX_MAX_ROWS=4))

        assert "Truncated" not in document.markdown
        assert document.metadata["truncated"] is False

    async def test_column_cap(self, override_settings: OverrideSettings) -> None:
        rows = [tuple(f"c{i}" for i in range(6)), tuple(range(6))]

        document = await _extract(_bytes(_book(("Wide", rows))), override_settings(XLSX_MAX_COLS=3))

        assert "| c0 | c1 | c2 |" in document.markdown
        assert "c3" not in document.markdown
        assert document.metadata["truncated"] is True

    async def test_sheet_cap(self, override_settings: OverrideSettings) -> None:
        sheets = [(f"S{n}", [("k", "v"), ("x", n)]) for n in range(1, 5)]

        document = await _extract(_bytes(_book(*sheets)), override_settings(XLSX_MAX_SHEETS=2))

        assert "## S2" in document.markdown and "## S3" not in document.markdown
        assert (document.metadata["sheets_extracted"], document.metadata["truncated"]) == (2, True)

    # ---- metadata ---- #

    async def test_workbook_properties(self) -> None:
        workbook = _book(("Data", [("k", "v"), ("a", 1)]))
        props = workbook.properties
        props.title, props.subject, props.creator, props.language = (
            "Fleet report",
            "Q2 numbers",
            "Dana",
            "en",
        )

        document = await _extract(_bytes(workbook))

        assert (document.title, document.description, document.lang) == (
            "Fleet report",
            "Q2 numbers",
            "en",
        )
        assert document.metadata["author"] == "Dana"

    # ---- failures ---- #

    async def test_corrupt_xlsx(self) -> None:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("[Content_Types].xml", "<Types/>")
            archive.writestr("xl/workbook.xml", "<broken")

        with pytest.raises(ExtractionError) as exc:
            await _extract(buffer.getvalue())

        assert (exc.value.reason, exc.value.detail) == (PageReason.EXTRACTION_FAILED, "corrupt")

    async def test_zip_bomb_guard_applies(self, override_settings: OverrideSettings) -> None:
        body = _bytes(_book(("Data", [("x" * 5000,)])))

        with pytest.raises(DocumentTooLargeError):
            await _extract(body, override_settings(OOXML_MAX_UNCOMPRESSED_BYTES=2000))

    # ---- integration ---- #

    async def test_dispatched_by_bytes_through_the_registry(self) -> None:
        body = (SITE_ROOT / "files" / "pricing.xlsx").read_bytes()
        source = ExtractionInput(url="https://acme.example/export", mime_type="", body=body)

        document = await ExtractorRegistry([XlsxExtractor()]).extract(source)

        assert document.extractor == "xlsx"

    async def test_fixture_xlsx_over_http(
        self, fetcher: HttpFetcher, fixture_site: FixtureSite
    ) -> None:
        result = await fetcher.fetch(fixture_site.url("/files/pricing.xlsx"))

        document = await XlsxExtractor().extract(ExtractionInput.from_fetch(result))

        assert result.content_kind is ContentKind.XLSX
        assert "| SNS-T1 | Torque sensor pack | 249 |" in document.markdown
