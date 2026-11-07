"""PPTX → Markdown with python-pptx: one section per slide.

Each visible slide becomes:

    <!-- slide 3 -->

    ## Slide title            (or "## Slide 3" when it has none)

    - body bullet             (body/content placeholders; nested by paragraph level)
    Free text box text        (other text shapes, as paragraphs)
    | table | … |             (tables; merged cells appear once)

    > Speaker notes text      (notes, when present)

- Shapes are read top-to-bottom, left-to-right; group shapes are recursed into, and chart
  titles are kept. Footer, date and slide-number placeholders are skipped (template).
- **Hidden slides are skipped** (drafts and backups); `metadata.hidden_skipped` counts them.
- Only the first `PPTX_MAX_SLIDES` visible slides are extracted (`metadata.truncated`).
- `<!-- slide N -->` mirrors the PDF page markers, so chunks can cite their slide.

Metadata: `title` (core properties, else the first slide's title), `lang` (core properties,
else the first run's `lang`), `description` (subject), author, created, modified, slide
counts. Hyperlinks (text runs and shape click actions) go to `outlinks`. The archive is
size-checked first (zip-bomb guard, `OOXML_MAX_UNCOMPRESSED_BYTES`).
"""

import io
import re
from collections.abc import Iterable, Iterator
from contextlib import suppress
from typing import Any

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER

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
from rag_engine.extraction.tables import render_table

_TITLE_TYPES = frozenset(
    {PP_PLACEHOLDER.TITLE, PP_PLACEHOLDER.CENTER_TITLE, PP_PLACEHOLDER.VERTICAL_TITLE}
)
_BULLET_TYPES = frozenset(
    {PP_PLACEHOLDER.BODY, PP_PLACEHOLDER.OBJECT, PP_PLACEHOLDER.VERTICAL_BODY,
     PP_PLACEHOLDER.VERTICAL_OBJECT}
)  # fmt: skip
_TEMPLATE_TYPES = frozenset(
    {PP_PLACEHOLDER.FOOTER, PP_PLACEHOLDER.DATE, PP_PLACEHOLDER.SLIDE_NUMBER,
     PP_PLACEHOLDER.HEADER}
)  # fmt: skip
_LANG = re.compile(r"^([a-z]{2,3})(?:[-_][a-z0-9]{1,8})*$", re.I)
_A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


def slide_marker(number: int) -> str:
    return f"<!-- slide {number} -->"


class PptxExtractor(BaseExtractor):
    name = "pptx"
    kind = ContentKind.PPTX
    mime_types = frozenset(
        {"application/vnd.openxmlformats-officedocument.presentationml.presentation"}
    )
    extensions = frozenset({".pptx"})

    def __init__(self, config: Settings = settings) -> None:
        self._max_uncompressed = config.OOXML_MAX_UNCOMPRESSED_BYTES
        self._max_slides = config.PPTX_MAX_SLIDES

    async def extract(self, source: ExtractionInput) -> ExtractedDocument:
        check_archive(source.body, self._max_uncompressed)
        return await self.run_sync(self._extract, source)

    def _extract(self, source: ExtractionInput) -> ExtractedDocument:
        try:
            prs = Presentation(io.BytesIO(source.body))
        except Exception as exc:  # PackageNotFoundError, KeyError for missing parts, …
            raise ExtractionError(f"cannot open PPTX: {exc}", detail="corrupt") from exc

        slides = list(prs.slides)
        visible = [(n, s) for n, s in enumerate(slides, 1) if s._element.get("show") != "0"]
        extracted = visible[: self._max_slides]
        sections: list[str] = []
        links: dict[str, None] = {}
        first_title: str | None = None
        for number, slide in extracted:
            title, section = _slide_section(number, slide, links)
            first_title = first_title or title
            if section:
                sections.append(section)
        if not sections:
            raise EmptyContentError("no text on any slide", detail="no_text")

        props = prs.core_properties
        metadata: dict[str, MetadataValue] = {
            "author": (props.author or "").strip() or None,
            "created": props.created.date().isoformat() if props.created else None,
            "modified": props.modified.date().isoformat() if props.modified else None,
            "slide_count": len(slides),
            "slides_extracted": len(extracted),
            "hidden_skipped": len(slides) - len(visible),
            "truncated": len(extracted) < len(visible),
        }
        return ExtractedDocument(
            markdown="\n\n".join(sections) + "\n",
            content_kind=ContentKind.PPTX,
            extractor=self.name,
            title=(props.title or "").strip() or first_title,
            lang=_language(props.language, (s for _, s in extracted)),
            description=(props.subject or "").strip() or None,
            metadata=metadata,
            outlinks=tuple(links),
        )


def _slide_section(number: int, slide: Any, links: dict[str, None]) -> tuple[str | None, str]:
    """(slide title, Markdown section). The section is "" for a slide with no text at all."""
    title: str | None = None
    blocks: list[str] = []
    for shape in _reading_order(slide.shapes):
        placeholder = _placeholder_type(shape)
        if placeholder in _TEMPLATE_TYPES:
            continue
        _collect_links(shape, links)
        if placeholder in _TITLE_TYPES and shape.has_text_frame:
            title = title or _clean(shape.text_frame.text) or None
            continue
        if getattr(shape, "has_table", False) and shape.has_table:
            if table := _table(shape.table):
                blocks.append(table)
        elif getattr(shape, "has_chart", False) and shape.has_chart:
            chart = shape.chart
            if chart.has_title and (text := _clean(chart.chart_title.text_frame.text)):
                blocks.append(f"Chart: {text}")
        elif shape.has_text_frame and (
            text := _text_frame(shape.text_frame, bullets=placeholder in _BULLET_TYPES)
        ):
            blocks.append(text)

    notes = ""
    if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
        lines = [_clean(p.text) for p in slide.notes_slide.notes_text_frame.paragraphs]
        notes = "\n".join(f"> {line}" for line in lines if line)
    if notes:
        blocks.append(notes)
    if title is None and not blocks:
        return None, ""
    heading = f"## {title}" if title else f"## Slide {number}"
    return title, "\n\n".join([slide_marker(number), heading, *blocks])


def _reading_order(shapes: Iterable[Any]) -> Iterator[Any]:
    """Shapes top-to-bottom, left-to-right; group shapes flattened in place."""
    ordered = sorted(shapes, key=lambda s: (s.top or 0, s.left or 0))
    for shape in ordered:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from _reading_order(shape.shapes)
        else:
            yield shape


def _placeholder_type(shape: Any) -> Any:
    if not shape.is_placeholder:
        return None
    try:
        return shape.placeholder_format.type
    except ValueError:  # malformed placeholder element
        return None


def _text_frame(frame: Any, *, bullets: bool) -> str:
    lines: list[str] = []
    for paragraph in frame.paragraphs:
        text = _clean(paragraph.text)
        if not text:
            continue
        if bullets:
            lines.append(f"{'   ' * min(paragraph.level, 8)}- {text}")
        else:
            lines.append(text)
    return "\n".join(lines) if bullets else "\n\n".join(lines)


def _table(table: Any) -> str:
    rows = [
        ["" if cell.is_spanned else cell.text for cell in row.cells]  # merged: shown once
        for row in table.rows
    ]
    return render_table(rows)


def _clean(text: str) -> str:
    # python-pptx renders soft line breaks as vertical tabs
    return " ".join(text.replace("\v", " ").split())


def _collect_links(shape: Any, links: dict[str, None]) -> None:
    candidates: list[str | None] = []
    with suppress(AttributeError, KeyError, ValueError):  # shapes without click actions
        candidates.append(shape.click_action.hyperlink.address)
    if shape.has_text_frame:
        for paragraph in shape.text_frame.paragraphs:
            candidates.extend(run.hyperlink.address for run in paragraph.runs)
    for address in candidates:
        if address and address.strip().lower().startswith(("http://", "https://")):
            links.setdefault(address.strip(), None)


def _language(core_language: str | None, slides: Iterable[Any]) -> str | None:
    candidates: list[str | None] = [core_language]
    for slide in slides:
        run_props = slide._element.find(f".//{_A_NS}rPr[@lang]")
        if run_props is not None:
            candidates.append(run_props.get("lang"))
            break
    for value in candidates:
        match = _LANG.match((value or "").strip())
        if match:
            return match.group(1).lower()
    return None
