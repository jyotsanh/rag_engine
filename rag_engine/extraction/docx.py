"""DOCX → Markdown with python-docx, in document order (paragraphs and tables interleaved).

- **Headings**: `Title` → `#`; `Heading N` (also custom styles *based on* a heading style)
  → N `#`s, capped at 6. When style names are localized ("Überschrift 1"), the style's
  outline level decides.
- **Lists**: paragraphs with Word numbering (direct or via the style). Bullet vs ordered
  comes from the numbering definition's format; nesting from `ilvl`. Styles named
  "List Number"/"List Bullet" are the fallback.
- **Inline**: bold → `**…**`, italic → `*…*` (adjacent runs with equal formatting merged).
  Hyperlink text is kept, and its URL goes to `outlinks`.
- **Code**: consecutive paragraphs in a code-like style (`Code`, `Source Code`,
  `HTML Preformatted`, …) become one fenced block. Quote styles → `> `.
- **Tables** → Markdown tables (first row = header). Merged cells appear once (repeats
  blank), `|` escaped, multi-paragraph cells joined with spaces.
- Page headers/footers, footnotes and comments are ignored (they are boilerplate).

Metadata: `title` (core properties, else the first heading), `lang` (core properties, else
the document's default run language), `description` (subject), author, created, modified.
The archive is size-checked first (zip-bomb guard, `OOXML_MAX_UNCOMPRESSED_BYTES`).
"""

import io
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.hyperlink import Hyperlink
from docx.text.paragraph import Paragraph

from rag_engine.config import Settings, settings
from rag_engine.domain import ContentKind
from rag_engine.extraction.base import BaseExtractor, ExtractedDocument, ExtractionInput
from rag_engine.extraction.exceptions import EmptyContentError, ExtractionError
from rag_engine.extraction.ooxml import check_archive

_HEADING_STYLE = re.compile(r"^heading\s*(\d)$", re.I)
_CODE_STYLE = re.compile(r"code|source|preformatted|verbatim|monospace", re.I)
_QUOTE_STYLE = re.compile(r"quote", re.I)
_ORDERED_STYLE = re.compile(r"list number", re.I)
_LIST_STYLE = re.compile(r"^list", re.I)
_LANG = re.compile(r"^([a-z]{2,3})(?:[-_][a-z0-9]{1,8})*$", re.I)
_BLANK_RUNS = re.compile(r"\n{3,}")
_MAX_STYLE_DEPTH = 10  # base-style chains are short; guards against cycles in bad files


@dataclass(frozen=True, slots=True)
class _ListInfo:
    level: int
    ordered: bool


class DocxExtractor(BaseExtractor):
    name = "docx"
    kind = ContentKind.DOCX
    mime_types = frozenset(
        {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    )
    extensions = frozenset({".docx"})

    def __init__(self, config: Settings = settings) -> None:
        self._max_uncompressed = config.OOXML_MAX_UNCOMPRESSED_BYTES

    async def extract(self, source: ExtractionInput) -> ExtractedDocument:
        check_archive(source.body, self._max_uncompressed)
        return await self.run_sync(self._extract, source)

    def _extract(self, source: ExtractionInput) -> ExtractedDocument:
        try:
            doc = Document(io.BytesIO(source.body))
        except Exception as exc:  # PackageNotFoundError, KeyError for missing parts, …
            raise ExtractionError(f"cannot open DOCX: {exc}", detail="corrupt") from exc

        numbering = _numbering_formats(doc)
        blocks: list[str] = []
        code: list[str] = []
        first_heading: str | None = None

        def flush_code() -> None:
            if code:
                blocks.append("```\n" + "\n".join(code) + "\n```")
                code.clear()

        for item in doc.iter_inner_content():
            if isinstance(item, Table):
                flush_code()
                if table := _table_markdown(item):
                    blocks.append(table)
                continue
            style_name = item.style.name if item.style is not None else ""
            if _CODE_STYLE.search(style_name):
                code.append(item.text)
                continue
            flush_code()
            text = _inline_markdown(item)
            if not text:
                continue
            level = _heading_level(item)
            if level is not None:
                blocks.append(f"{'#' * level} {_plain(item)}")
                first_heading = first_heading or _plain(item)
                continue
            listing = _list_info(item, numbering)
            if listing is not None:
                marker = "1." if listing.ordered else "-"
                # 3 spaces per level nest under both "- " and "1. " parents (CommonMark)
                blocks.append(f"{'   ' * listing.level}{marker} {text}")
            elif _QUOTE_STYLE.search(style_name):
                blocks.append(f"> {text}")
            else:
                blocks.append(text)
        flush_code()

        markdown = _join(blocks)
        if not markdown:
            raise EmptyContentError("no text in the DOCX", detail="no_text")
        props = doc.core_properties
        return ExtractedDocument(
            markdown=markdown,
            content_kind=ContentKind.DOCX,
            extractor=self.name,
            title=(props.title or "").strip() or first_heading,
            lang=_language(props.language, doc),
            description=(props.subject or "").strip() or None,
            metadata={
                "author": (props.author or "").strip() or None,
                "created": props.created.date().isoformat() if props.created else None,
                "modified": props.modified.date().isoformat() if props.modified else None,
            },
            outlinks=tuple(dict.fromkeys(_hyperlinks(doc))),
        )


# ---- paragraphs ---- #


def _styles(paragraph: Paragraph) -> Iterator[Any]:
    style = paragraph.style
    for _ in range(_MAX_STYLE_DEPTH):
        if style is None:
            return
        yield style
        style = style.base_style


def _heading_level(paragraph: Paragraph) -> int | None:
    for style in _styles(paragraph):
        name = (style.name or "").strip()
        if name.lower() == "title":
            return 1
        match = _HEADING_STYLE.match(name)
        if match:
            return min(int(match.group(1)), 6)
    # localized templates: the outline level (0 = level 1) is language-independent
    for element in (paragraph._p.pPr, *(s.element.pPr for s in _styles(paragraph))):
        outline = element.find(qn("w:outlineLvl")) if element is not None else None
        if outline is not None:
            level = int(outline.get(qn("w:val"), "9"))
            return min(level + 1, 6) if level < 9 else None  # 9 = body text
    return None


def _list_info(paragraph: Paragraph, numbering: dict[tuple[str, int], str]) -> _ListInfo | None:
    num_pr = None
    for element in (paragraph._p.pPr, *(s.element.pPr for s in _styles(paragraph))):
        if element is not None and (num_pr := element.find(qn("w:numPr"))) is not None:
            break
    style_name = paragraph.style.name if paragraph.style is not None else ""
    if num_pr is None:
        if not _LIST_STYLE.match(style_name):
            return None
        return _ListInfo(_style_level(style_name), bool(_ORDERED_STYLE.search(style_name)))
    ilvl = num_pr.find(qn("w:ilvl"))
    num_id = num_pr.find(qn("w:numId"))
    level = int(ilvl.get(qn("w:val"), "0")) if ilvl is not None else _style_level(style_name)
    numbering_id = num_id.get(qn("w:val")) if num_id is not None else None
    if numbering_id == "0":
        return None  # numbering explicitly removed on this paragraph
    fmt = numbering.get((numbering_id or "", level))
    ordered = (
        fmt not in (None, "bullet", "none") if fmt else bool(_ORDERED_STYLE.search(style_name))
    )
    return _ListInfo(min(level, 8), ordered)


def _style_level(style_name: str) -> int:
    """`List Bullet 2` → level 1."""
    match = re.search(r"(\d)\s*$", style_name)
    return int(match.group(1)) - 1 if match else 0


def _numbering_formats(doc: Any) -> dict[tuple[str, int], str]:
    """(numId, ilvl) → numFmt ("bullet", "decimal", …) from numbering.xml."""
    try:
        root = doc.part.numbering_part.element
    except KeyError, NotImplementedError, AttributeError:
        return {}
    abstract: dict[str, dict[int, str]] = {}
    for definition in root.findall(qn("w:abstractNum")):
        levels: dict[int, str] = {}
        for lvl in definition.findall(qn("w:lvl")):
            fmt = lvl.find(qn("w:numFmt"))
            levels[int(lvl.get(qn("w:ilvl"), "0"))] = (
                fmt.get(qn("w:val"), "decimal") if fmt is not None else "decimal"
            )
        abstract[definition.get(qn("w:abstractNumId"), "")] = levels
    formats: dict[tuple[str, int], str] = {}
    for num in root.findall(qn("w:num")):
        ref = num.find(qn("w:abstractNumId"))
        if ref is None:
            continue
        for level, fmt in abstract.get(ref.get(qn("w:val"), ""), {}).items():
            formats[(num.get(qn("w:numId"), ""), level)] = fmt
    return formats


def _inline_markdown(paragraph: Paragraph) -> str:
    """Paragraph text with bold/italic; adjacent runs with equal formatting are merged."""
    pieces: list[tuple[str, bool, bool]] = []
    for item in paragraph.iter_inner_content():
        runs = item.runs if isinstance(item, Hyperlink) else [item]
        for run in runs:
            if run.text:
                pieces.append((run.text, bool(run.bold), bool(run.italic)))
    merged: list[tuple[str, bool, bool]] = []
    for text, bold, italic in pieces:
        if merged and merged[-1][1:] == (bold, italic):
            merged[-1] = (merged[-1][0] + text, bold, italic)
        else:
            merged.append((text, bold, italic))
    out = []
    for text, bold, italic in merged:
        core = text.strip()
        if not core or not (bold or italic):
            out.append(text)
            continue
        mark = "**" if bold and not italic else "*" if italic and not bold else "***"
        lead, trail = text[: len(text) - len(text.lstrip())], text[len(text.rstrip()) :]
        out.append(f"{lead}{mark}{core}{mark}{trail}")
    return " ".join("".join(out).split())


def _plain(paragraph: Paragraph) -> str:
    return " ".join(paragraph.text.split())


def _hyperlinks(doc: Any) -> Iterator[str]:
    for paragraph in _all_paragraphs(doc):
        for item in paragraph.iter_inner_content():
            if isinstance(item, Hyperlink):
                url = (item.url or "").strip()
                if url.lower().startswith(("http://", "https://")):
                    yield url


def _all_paragraphs(doc: Any) -> Iterator[Paragraph]:
    yield from doc.paragraphs
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                yield from cell.paragraphs


# ---- tables ---- #


def _table_markdown(table: Table) -> str:
    rows: list[list[str]] = []
    above: dict[int, Any] = {}
    for row in table.rows:
        cells: list[str] = []
        previous = None
        for column, cell in enumerate(row.cells):
            tc = cell._tc
            repeated = tc is previous or above.get(column) is tc  # merged span
            text = "" if repeated else _cell_text(cell)
            cells.append(text)
            previous, above[column] = tc, tc
        rows.append(cells)
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _cell_text(cell: Any) -> str:
    text = " ".join(" ".join(p.text.split()) for p in cell.paragraphs if p.text.strip())
    for nested in cell.tables:  # nested tables are flattened into the cell
        for row in nested.rows:
            text += " " + " ".join(_cell_text(c) for c in row.cells)
    return text.strip().replace("|", "\\|")


# ---- document level ---- #


def _language(core_language: str | None, doc: Any) -> str | None:
    candidates = [core_language]
    try:
        defaults = doc.styles.element.find(qn("w:docDefaults"))
        lang = defaults.find(f"{qn('w:rPrDefault')}/{qn('w:rPr')}/{qn('w:lang')}")
        candidates.append(lang.get(qn("w:val")) if lang is not None else None)
    except AttributeError:
        pass
    for value in candidates:
        match = _LANG.match((value or "").strip())
        if match:
            return match.group(1).lower()
    return None


def _join(blocks: list[str]) -> str:
    """Blank lines between blocks, except between consecutive items of one list."""
    out: list[str] = []
    for block in blocks:
        if out:
            list_item = re.match(r"^\s*(-|1\.) ", block)
            previous_list = re.match(r"^\s*(-|1\.) ", out[-1])
            out.append("\n" if list_item and previous_list else "\n\n")
        out.append(block)
    text = _BLANK_RUNS.sub("\n\n", "".join(out)).strip()
    return text + "\n" if text else ""
