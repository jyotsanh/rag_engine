"""The structure chunker: Markdown structure first, token budget second (Step 57).

Registered as `ChunkerType.STRUCTURE` (the default `CHUNKER`). For each section of the
parsed page (Step 56), in order:

1. **Units.** Every block becomes one unit when it fits, so paragraphs, tables, code and
   lists stay whole. A prose block (paragraph, quote, raw HTML) over `target_tokens` is cut
   into sentences, then lines, then words, then (a single "word" longer than the cap, e.g. a
   base64 blob) characters, recursively, until each unit fits the cap. An **atomic** block
   (table, code, list) is only cut when it is over the cap, and then along its own seams:
   - table: rows, with the header and separator row repeated on every piece;
   - fenced code: lines, every piece re-wrapped in the original fence (an unclosed fence
     gets a matching closing fence); indented code: lines;
   - list: top-level items (an item over the cap falls back to prose cutting).
2. **Packing.** Units are packed greedily up to `target_tokens`. A unit that is alone over
   the target (an atomic block between target and cap) becomes its own chunk. Sections are
   never merged: a chunk has exactly one heading path, the section's.
3. **Overlap.** Only where a cut falls *inside* a prose block (a long paragraph split
   between sentences or lines), the next chunk begins with the last whole sentences of the
   previous one, up to `overlap_tokens`: that is where context is lost. Block, paragraph
   and section boundaries are natural breaks and are not overlapped, and neither is atomic
   content.

**Budgets.** `target_tokens` is measured on the chunk text. The hard cap is measured on
what is embedded: the room the context prefix takes (`embed_text`, Step 58) for the
section's heading path is reserved, so `text + context <= max_tokens` always holds and the
base class's `ChunkTooLargeError` means a bug.

`page_number` of a chunk is the location (page / slide / sheet) of its first own unit,
else the section's. Heading lines are not repeated in `text` (`heading_path` carries them).
"""

import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace

from rag_engine.chunking.base import BaseChunker, ChunkInput, ChunkPiece
from rag_engine.chunking.exceptions import ChunkingError
from rag_engine.chunking.factory import register_chunker
from rag_engine.chunking.markdown_parser import Block, BlockKind, MarkdownParser, Section
from rag_engine.config import ChunkerType

# a sentence ends at . ! ? or an ellipsis (or the full-width CJK stops), optionally followed
# by closing quotes/brackets, then whitespace; the next sentence starts at the next non-space
_CLOSERS = "\"')]\u201d\u2019"  # incl. right double/single quotation marks
_OPENERS = "(\"'[\u201c\u2018"  # incl. left double/single quotation marks
_CJK_STOPS = "\u3002\uff01\uff1f"  # ideographic full stop, full-width ! and ?
_SENTENCE_END = re.compile(rf"[.!?\u2026{_CJK_STOPS}][{re.escape(_CLOSERS)}]*(?=\s+\S)")
_CJK_END = re.compile(rf"[{_CJK_STOPS}](?=\S)")
# a period after these words (or a lone letter: initials) does not end a sentence
_ABBREVIATIONS = frozenset(
    [
        "e.g",
        "i.e",
        "etc",
        "vs",
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "no",
        "fig",
        "figs",
        "eq",
        "approx",
        "inc",
        "ltd",
        "co",
        "corp",
        "dept",
        "est",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "al",
        "cf",
        "p",
        "pp",
        "vol",
    ]
)
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


@dataclass(frozen=True, slots=True)
class _Unit:
    text: str
    block: int  # index of the source block (units of one block join without a blank line)
    joiner: str  # how this unit attaches to the previous unit of the same block
    location: int | None
    prose: bool  # sentences of prose can be overlapped; atomic content never is
    overlap: bool = False  # carried over from the previous chunk


def sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping abbreviations and initials together."""
    cuts: list[int] = []
    for match in _SENTENCE_END.finditer(text):
        if text[match.start()] == ".":
            word = text[: match.start()].rsplit(None, 1)[-1] if text[: match.start()] else ""
            word = word.lstrip(_OPENERS).lower()
            if word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha()):
                continue
        cuts.append(match.end())
    cuts.extend(m.end() for m in _CJK_END.finditer(text))
    pieces: list[str] = []
    start = 0
    for cut in sorted(set(cuts)):
        piece = text[start:cut].strip()
        if piece:
            pieces.append(piece)
        start = cut
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces


@register_chunker(ChunkerType.STRUCTURE)
class StructureChunker(BaseChunker):
    name = "structure"

    _parser = MarkdownParser()

    def split(self, source: ChunkInput) -> Iterable[ChunkPiece]:
        for section in self._parser.parse(source.markdown).sections:
            if not section.blocks:
                continue  # a heading with no content: its title lives on in child paths
            limit = self._limit(source, section)
            units = [u for i, b in enumerate(section.blocks) for u in self._units(i, b, limit)]
            yield from self._pack(units, section, limit)

    # ---- budgets ---- #

    def _count(self, text: str) -> int:
        return self.tokenizer.count(text)

    def _limit(self, source: ChunkInput, section: Section) -> int:
        """Tokens a chunk's text may use once the embed-text context is added."""
        probe = "x"
        context = ChunkPiece(probe, section.heading_path)
        reserve = self._count(self.embed_text(source, context, probe)) - self._count(probe)
        limit = self.config.max_tokens - max(reserve, 0)
        if limit < 1:
            raise ChunkingError(
                f"{source.url}: the embed-text context alone takes {reserve} tokens "
                f"(cap {self.config.max_tokens}) for heading path {section.heading_path!r}"
            )
        return limit

    # ---- units ---- #

    def _units(self, index: int, block: Block, limit: int) -> list[_Unit]:
        def unit(text: str, joiner: str = "\n", *, prose: bool = False) -> _Unit:
            return _Unit(text, index, joiner, block.location, prose)

        tokens = self._count(block.text)
        if not block.atomic:
            if tokens <= min(self.config.target_tokens, limit):
                return [unit(block.text, prose=True)]
            return [unit(t, j, prose=True) for t, j in self._prose(block.text, limit)]
        if tokens <= limit:
            return [unit(block.text)]
        match block.kind:
            case BlockKind.TABLE:
                pieces = self._table(block, limit)
            case BlockKind.CODE:
                pieces = self._code(block, limit)
            case _:
                pieces = self._list(block, limit)
        return [unit(text) for text in pieces]

    def _prose(self, text: str, limit: int) -> list[tuple[str, str]]:
        """(piece, joiner) with every piece within `limit`: sentences → lines → words → chars."""
        splitters: list[tuple[Callable[[str], list[str]], str]] = [
            (sentences, " "),
            (lambda t: [line for line in t.split("\n") if line.strip()], "\n"),
        ]
        return self._fit(text, limit, splitters)

    def _fit(
        self, text: str, limit: int, splitters: list[tuple[Callable[[str], list[str]], str]]
    ) -> list[tuple[str, str]]:
        for position, (split, joiner) in enumerate(splitters):
            parts = split(text)
            if len(parts) > 1:
                out: list[tuple[str, str]] = []
                for part in parts:
                    if self._count(part) <= limit:
                        out.append((part, joiner))
                    else:
                        out.extend(self._fit(part, limit, splitters[position + 1 :]))
                return out
        return [(piece, " ") for piece in self._words(text, limit)]

    def _words(self, text: str, limit: int) -> list[str]:
        """Greedy word groups within `limit`; a single word over it is cut by characters."""
        groups: list[str] = []
        current = ""
        for word in text.split():
            candidate = f"{current} {word}" if current else word
            if self._count(candidate) <= limit:
                current = candidate
                continue
            if current:
                groups.append(current)
            if self._count(word) <= limit:
                current = word
            else:
                *full, current = self._characters(word, limit)
                groups.extend(full)
        if current:
            groups.append(current)
        return groups

    def _characters(self, text: str, limit: int) -> list[str]:
        """Cut text with no usable boundary into the longest prefixes that fit `limit`."""
        pieces: list[str] = []
        while text:
            low, high = 1, len(text)
            while low < high:  # longest prefix within the limit (token counts are monotonic)
                middle = (low + high + 1) // 2
                if self._count(text[:middle]) <= limit:
                    low = middle
                else:
                    high = middle - 1
            pieces.append(text[:low])
            text = text[low:]
        return pieces

    def _group(self, prefix: str, items: list[str], suffix: str, limit: int) -> list[str]:
        """Pack `items` (joined by newlines) between `prefix` and `suffix` within `limit`.

        An item that cannot fit even alone is returned as prose pieces without the wrapper.
        """

        def wrap(body: list[str]) -> str:
            return "\n".join(part for part in (prefix, *body, suffix) if part)

        groups: list[str] = []
        current: list[str] = []
        for item in items:
            if self._count(wrap([*current, item])) <= limit:
                current.append(item)
                continue
            if current:
                groups.append(wrap(current))
                current = []
            if self._count(wrap([item])) <= limit:
                current = [item]
            else:  # one row / line / item too large on its own
                groups.extend(piece for piece, _ in self._prose(item, limit))
        if current:
            groups.append(wrap(current))
        return groups

    def _table(self, block: Block, limit: int) -> list[str]:
        header = block.header or ""
        if self._count(header) >= limit:  # a header that can't carry any row: plain text
            return [piece for piece, _ in self._prose(block.text, limit)]
        return self._group(header, list(block.parts), "", limit)

    def _code(self, block: Block, limit: int) -> list[str]:
        if block.header is None:  # indented code
            return self._group("", list(block.parts), "", limit)
        fence = _FENCE.match(block.header)
        footer = block.footer or (fence.group(1) if fence else "```")
        lines = list(block.parts)
        room = limit - self._count(f"{block.header}\n\n{footer}")
        if room < 1:  # fences that leave no room for code: plain text
            return [piece for piece, _ in self._prose(block.text, limit)]
        wrapped: list[str] = []
        for line in lines:  # a line too long even alone is cut, staying inside the fence
            wrapped.extend(self._characters(line, room) if self._count(line) > room else [line])
        return self._group(block.header, wrapped, footer, limit)

    def _list(self, block: Block, limit: int) -> list[str]:
        return self._group("", list(block.parts), "", limit)

    # ---- packing ---- #

    @staticmethod
    def _join(units: list[_Unit]) -> str:
        text = ""
        for i, unit in enumerate(units):
            if i == 0:
                text = unit.text
            elif unit.block == units[i - 1].block:
                text += unit.joiner + unit.text
            else:
                text += "\n\n" + unit.text
        return text

    def _pack(self, units: list[_Unit], section: Section, limit: int) -> Iterator[ChunkPiece]:
        target = min(self.config.target_tokens, limit)
        current: list[_Unit] = []

        def piece(chunk: list[_Unit]) -> ChunkPiece:
            own = next((u for u in chunk if not u.overlap), chunk[0])
            location = own.location if own.location is not None else section.location
            return ChunkPiece(self._join(chunk), section.heading_path, location)

        for unit in units:
            if not current or self._count(self._join([*current, unit])) <= target:
                current.append(unit)
                continue
            yield piece(current)
            carried = self._overlap(current, unit, limit)
            current = [*carried, unit]
        if current:
            yield piece(current)

    def _overlap(self, previous: list[_Unit], unit: _Unit, limit: int) -> list[_Unit]:
        """The previous chunk's last sentences (≤ overlap_tokens) when a prose block was cut."""
        budget = self.config.overlap_tokens
        last = previous[-1]
        if budget <= 0 or not (last.prose and unit.prose and last.block == unit.block):
            return []
        carried: list[_Unit] = []
        for sentence in reversed(sentences(last.text)):
            candidate = [replace(last, text=sentence, joiner=" ", overlap=True), *carried]
            if self._count(self._join(candidate)) > budget:
                break
            carried = candidate
        if carried and self._count(self._join([*carried, unit])) > limit:
            return []  # never let the overlap push a chunk over the cap
        return carried
