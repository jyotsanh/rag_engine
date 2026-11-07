"""Markdown structure parser: a page's Markdown → sections by heading → blocks (Step 56).

The chunker (Step 57) cuts along this structure instead of along characters:

- **Sections** follow the heading hierarchy. Each carries its `heading_path` (the titles of
  the headings above it, outermost first, as plain text); content before the first heading
  is the preamble (empty path). A deeper heading nests, an equal or shallower one closes the
  open sections (levels may be skipped: `# A` then `### B` is `A > B`).
- **Blocks** are the top-level Markdown blocks of a section, each with its exact source
  text (so tables and code keep their formatting). Tables, code (fenced or indented) and
  lists are **atomic**: the splitter keeps them whole unless they are oversized, and then
  cuts them along `parts` (table rows under a repeated `header`, code lines inside the
  re-opened fence, list items). Paragraphs, quotes and raw HTML are prose.
- **Location markers** from the extractors, `<!-- page N -->` (PDF), `<!-- slide N -->`
  (PPTX) and `<!-- sheet N -->` (XLSX), are not content: they set the `location` of the
  blocks and headings that follow. Other HTML comments and thematic breaks are dropped.

Parsing uses markdown-it-py (CommonMark + GFM tables and strikethrough). Only top-level
tokens and their source line ranges (`map`) are used, so a `#` inside code or a table cell
is never mistaken for a heading.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from markdown_it import MarkdownIt
from markdown_it.token import Token

_LOCATION = re.compile(r"^\s*<!--\s*(page|slide|sheet)\s+(\d+)\s*-->\s*$", re.I)
_COMMENT = re.compile(r"^\s*<!--.*-->\s*$", re.S)


class BlockKind(StrEnum):
    PARAGRAPH = "paragraph"
    QUOTE = "quote"
    HTML = "html"  # raw HTML that is not a comment
    TABLE = "table"
    CODE = "code"
    LIST = "list"

    @property
    def atomic(self) -> bool:
        return self in _ATOMIC


_ATOMIC = frozenset({BlockKind.TABLE, BlockKind.CODE, BlockKind.LIST})


@dataclass(frozen=True, slots=True)
class Block:
    kind: BlockKind
    text: str  # the block's Markdown source, blank lines around it trimmed
    location: int | None = None  # page/slide/sheet number the block is on
    line: int = 0  # 0-based source line where it starts
    # how an oversized atomic block may be cut (Step 57): `header` + some `parts` + `footer`
    header: str | None = None  # table: header + separator rows; code: the opening fence
    footer: str | None = None  # code: the closing fence (None when the fence never closed)
    parts: tuple[str, ...] = ()  # table rows, code lines, list items (with nested content)
    info: str | None = None  # code: the fence's info string (language)

    @property
    def atomic(self) -> bool:
        return self.kind.atomic


@dataclass(slots=True)
class Section:
    heading_path: tuple[str, ...]  # () for the preamble
    level: int  # heading level 1-6; 0 for the preamble
    heading: str | None = None  # plain-text title
    heading_markdown: str | None = None  # the heading's source line(s)
    location: int | None = None  # where the heading is
    blocks: list[Block] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    sections: tuple[Section, ...]
    location_kind: str | None = None  # "page" / "slide" / "sheet" if markers were present

    @property
    def blocks(self) -> list[Block]:
        return [block for section in self.sections for block in section.blocks]


def _parser() -> MarkdownIt:
    return MarkdownIt("commonmark").enable(["table", "strikethrough"])


_MD = _parser()


def heading_text(inline: Token) -> str:
    """A heading's plain text: markup, link targets and images' URLs dropped."""
    parts: list[str] = []
    for child in inline.children or ():
        if child.type in ("text", "code_inline"):
            parts.append(child.content)
        elif child.type == "image":
            parts.append(child.content)  # the alt text
        elif child.type in ("softbreak", "hardbreak"):
            parts.append(" ")
    return " ".join("".join(parts).split())


def _slice(lines: Sequence[str], start: int, end: int) -> str:
    chunk = list(lines[start:end])
    while chunk and not chunk[0].strip():
        chunk.pop(0)
    while chunk and not chunk[-1].strip():
        chunk.pop()
    return "\n".join(chunk)


def _closing(token: Token, tokens: Sequence[Token], index: int) -> int:
    """Index of the token closing the block opened at `index`."""
    depth = 0
    for position in range(index, len(tokens)):
        nesting = tokens[position].nesting
        depth += nesting
        if depth == 0:
            return position
    raise ValueError(f"unclosed {token.type}")  # markdown-it always closes its blocks


class MarkdownParser:
    """`parse(markdown) -> ParsedDocument`; stateless and reusable."""

    def parse(self, markdown: str) -> ParsedDocument:
        source = markdown.replace("\r\n", "\n").replace("\r", "\n")
        lines = source.split("\n")
        tokens = _MD.parse(source)

        sections: list[Section] = [Section(heading_path=(), level=0)]
        stack: list[tuple[int, str]] = []  # open headings: (level, title)
        location: int | None = None
        location_kind: str | None = None

        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token.level != 0 or token.map is None:
                index += 1
                continue
            start, end = token.map
            close = _closing(token, tokens, index) if token.nesting == 1 else index

            if token.type == "heading_open":
                level = int(token.tag[1])
                title = heading_text(tokens[index + 1])
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                sections.append(
                    Section(
                        heading_path=tuple(t for _, t in stack),
                        level=level,
                        heading=title,
                        heading_markdown=_slice(lines, start, end),
                        location=location,
                    )
                )
            elif token.type == "html_block":
                text = _slice(lines, start, end)
                if marker := _LOCATION.match(text):
                    location_kind = location_kind or marker.group(1).lower()
                    location = int(marker.group(2))
                elif text and not _COMMENT.match(text):
                    sections[-1].blocks.append(
                        Block(BlockKind.HTML, text, location=location, line=start)
                    )
            elif token.type != "hr":
                block = self._block(token, tokens, index, close, lines, location)
                if block is not None and block.text:
                    sections[-1].blocks.append(block)
            index = close + 1

        kept = tuple(s for s in sections if s.level > 0 or s.blocks)
        return ParsedDocument(sections=kept, location_kind=location_kind)

    def _block(
        self,
        token: Token,
        tokens: Sequence[Token],
        index: int,
        close: int,
        lines: Sequence[str],
        location: int | None,
    ) -> Block | None:
        if token.map is None:
            return None
        start, end = token.map
        text = _slice(lines, start, end)
        match token.type:
            case "paragraph_open":
                return Block(BlockKind.PARAGRAPH, text, location=location, line=start)
            case "blockquote_open":
                return Block(BlockKind.QUOTE, text, location=location, line=start)
            case "table_open":
                rows = text.split("\n")
                return Block(
                    BlockKind.TABLE,
                    text,
                    location=location,
                    line=start,
                    header="\n".join(rows[:2]),
                    parts=tuple(rows[2:]),
                )
            case "fence":
                block_lines = text.split("\n")
                opening = block_lines[0]
                last = block_lines[-1].strip() if len(block_lines) > 1 else ""
                closed = last.startswith(token.markup) and set(last) <= set(token.markup[0])
                return Block(
                    BlockKind.CODE,
                    text,
                    location=location,
                    line=start,
                    header=opening,
                    footer=block_lines[-1] if closed else None,
                    parts=tuple(token.content.rstrip("\n").split("\n")),
                    info=token.info.strip() or None,
                )
            case "code_block":  # indented code: no fences to repeat
                return Block(
                    BlockKind.CODE,
                    text,
                    location=location,
                    line=start,
                    parts=tuple(text.split("\n")),
                )
            case "bullet_list_open" | "ordered_list_open":
                items = tuple(
                    _slice(lines, *item.map)
                    for item in tokens[index + 1 : close]
                    if item.type == "list_item_open" and item.level == 1 and item.map
                )
                return Block(BlockKind.LIST, text, location=location, line=start, parts=items)
            case _:
                return None  # nothing else exists at the top level of CommonMark + tables


def parse_markdown(markdown: str) -> ParsedDocument:
    return MarkdownParser().parse(markdown)
