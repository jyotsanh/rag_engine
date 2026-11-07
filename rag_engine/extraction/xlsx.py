"""XLSX → Markdown with openpyxl (read-only, streaming): one table per visible sheet.

    <!-- sheet N -->

    ## Sheet name

    Caption text              (a lone title row above the real header)

    | Header | … |
    |---|---|
    | value  | … |

    *Truncated: showing the first 2000 rows and 50 columns.*   (only when capped)

- Values are the **cached results** Excel stored (`data_only=True`), not formulas. A file
  written by a library without cached values shows those formula cells empty.
- Formatting: integral floats → `1899`, other floats → up to 10 significant digits,
  percent-formatted numbers → `5%`, dates → ISO (`2026-05-01`, time kept when not
  midnight), booleans → `TRUE`/`FALSE`.
- Entirely empty rows and columns are dropped (sheets often carry formatting far past
  their data).
- Hidden and very-hidden sheets are skipped (`metadata.hidden_skipped`); chart sheets
  carry no cells.
- Caps: `XLSX_MAX_SHEETS` visible sheets, `XLSX_MAX_ROWS` rows and `XLSX_MAX_COLS` columns
  per sheet. One extra row/column is read to detect truncation, which is noted in the
  Markdown and counted in `metadata.truncated_sheets`.
- `<!-- sheet N -->` mirrors the PDF/PPTX markers so chunks can cite their sheet.

Metadata: `title`/`description`/`lang` from the workbook properties; author, created,
modified, sheet counts. Hyperlinks are not available in read-only mode, so there are no
outlinks. The archive is size-checked first (zip-bomb guard).
"""

import io
from datetime import date, datetime, time
from typing import Any

from openpyxl import load_workbook

from rag_engine.config import Settings, settings
from rag_engine.domain import ContentKind
from rag_engine.extraction.base import (
    BaseExtractor,
    ExtractedDocument,
    ExtractionInput,
    MetadataValue,
)
from rag_engine.extraction.exceptions import EmptyContentError, ExtractionError
from rag_engine.extraction.ooxml import check_archive
from rag_engine.extraction.tables import cell_text, render_table


def sheet_marker(number: int) -> str:
    return f"<!-- sheet {number} -->"


class XlsxExtractor(BaseExtractor):
    name = "xlsx"
    kind = ContentKind.XLSX
    mime_types = frozenset({"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"})
    extensions = frozenset({".xlsx"})

    def __init__(self, config: Settings = settings) -> None:
        self._max_uncompressed = config.OOXML_MAX_UNCOMPRESSED_BYTES
        self._max_sheets = config.XLSX_MAX_SHEETS
        self._max_rows = config.XLSX_MAX_ROWS
        self._max_cols = config.XLSX_MAX_COLS

    async def extract(self, source: ExtractionInput) -> ExtractedDocument:
        check_archive(source.body, self._max_uncompressed)
        return await self.run_sync(self._extract, source)

    def _extract(self, source: ExtractionInput) -> ExtractedDocument:
        try:
            workbook = load_workbook(io.BytesIO(source.body), read_only=True, data_only=True)
        except Exception as exc:  # InvalidFileException, KeyError for missing parts, …
            raise ExtractionError(f"cannot open XLSX: {exc}", detail="corrupt") from exc
        try:
            return self._convert(workbook)
        finally:
            workbook.close()

    def _convert(self, workbook: Any) -> ExtractedDocument:
        sheets = list(workbook.worksheets)
        visible = [s for s in sheets if s.sheet_state == "visible"]
        extracted = visible[: self._max_sheets]
        sections: list[str] = []
        truncated_sheets = 0
        for number, sheet in enumerate(extracted, 1):
            section, truncated = self._sheet_section(number, sheet)
            truncated_sheets += truncated
            if section:
                sections.append(section)
        if not sections:
            raise EmptyContentError("no cell values in any visible sheet", detail="no_text")

        props = workbook.properties
        metadata: dict[str, MetadataValue] = {
            "author": (props.creator or "").strip() or None,
            "created": props.created.date().isoformat() if props.created else None,
            "modified": props.modified.date().isoformat() if props.modified else None,
            "sheet_count": len(sheets),
            "sheets_extracted": len(extracted),
            "hidden_skipped": len(sheets) - len(visible),
            "truncated_sheets": truncated_sheets,
            "truncated": truncated_sheets > 0 or len(extracted) < len(visible),
        }
        return ExtractedDocument(
            markdown="\n\n".join(sections) + "\n",
            content_kind=ContentKind.XLSX,
            extractor=self.name,
            title=(props.title or "").strip() or None,
            lang=(props.language or "").strip() or None,
            description=(props.subject or props.description or "").strip() or None,
            metadata=metadata,
        )

    def _sheet_section(self, number: int, sheet: Any) -> tuple[str, bool]:
        rows: list[list[str]] = []
        more_rows = more_cols = False
        for index, row in enumerate(
            sheet.iter_rows(max_row=self._max_rows + 1, max_col=self._max_cols + 1)
        ):
            if index == self._max_rows:
                more_rows = any(_format(c) for c in row)
                break
            values = [_format(cell) for cell in row]
            if len(values) > self._max_cols:
                more_cols = more_cols or bool(values[self._max_cols])
                values = values[: self._max_cols]
            rows.append(values)
        rows = _trim(rows)
        if not rows:
            return "", False

        blocks = [sheet_marker(number), f"## {cell_text(sheet.title) or f'Sheet {number}'}"]
        filled = [sum(1 for v in row if v) for row in rows]
        if len(rows) > 1 and filled[0] == 1 and filled[1] > 1:
            blocks.append(cell_text(next(v for v in rows[0] if v)))  # title row → caption
            rows = _trim(rows[1:])
        blocks.append(render_table(rows))
        truncated = more_rows or more_cols
        if truncated:
            blocks.append(
                f"*Truncated: showing the first {self._max_rows} rows "
                f"and {self._max_cols} columns.*"
            )
        return "\n\n".join(blocks), truncated


def _format(cell: Any) -> str:
    value = getattr(cell, "value", None)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int | float):
        if "%" in (getattr(cell, "number_format", "") or ""):
            return f"{value * 100:.10g}%"
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return f"{value:.10g}" if isinstance(value, float) else str(value)
    if isinstance(value, datetime):
        return value.date().isoformat() if value.time() == time(0) else value.isoformat(" ")
    if isinstance(value, date | time):
        return value.isoformat()
    return " ".join(str(value).split())


def _trim(rows: list[list[str]]) -> list[list[str]]:
    """Drop entirely empty rows and entirely empty columns."""
    rows = [row for row in rows if any(row)]
    if not rows:
        return []
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    keep = [c for c in range(width) if any(row[c] for row in rows)]
    return [[row[c] for c in keep] for row in rows]
