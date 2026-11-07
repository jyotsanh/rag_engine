"""Cross-page boilerplate: blocks repeated on many pages of one crawl job are template.

Per-page extraction (trafilatura) misses chrome that sits *inside* the content area on
some templates: cookie banners, "related articles" lists, shared notices. Across a whole
site it is obvious: the same block appears on most pages. `BoilerplateIndex` counts, per
job, on how many **distinct** pages each Markdown unit appears, and strips the frequent
ones.

Units: a paragraph, a table, or **one list item** (navigation lists differ per page by an
"active" item, but their other items repeat exactly). Headings and code fences are never
units: a heading is structure (Step 45 prunes it if its section empties), and code
repeated across docs pages is usually real content (install commands).

Matching is on a normalized form (NFKC, casefold, Markdown punctuation and list markers
ignored, whitespace collapsed), hashed to 8 bytes.

A unit is boilerplate when the index holds at least `min_pages` distinct pages and the
unit appears on at least `min_ratio` of them (and on two or more). Distinct means distinct
*content*: twenty URLs serving one page count once, so duplicates can't turn real content
into "boilerplate".

**Keep one copy**: a boilerplate unit with at least `keep_one_min_chars` normalized
characters (a shared warranty paragraph, not a nav link) is kept on the first page that
contained it (the crawl adds hub pages first), so the information stays searchable once.
Shorter units (nav items, cookie banners) are removed everywhere.

The index lives in memory for one job. It can always be rebuilt from the stored page
Markdown, so a resumed job loses nothing.
"""

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from rag_engine.config import Settings, settings

_FENCE = re.compile(r"^\s{0,3}(```|~~~)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}(\s|$)")
_LIST_ITEM = re.compile(r"^\s{0,3}(?:[-*+]|\d{1,9}[.)])\s+")
_MARKUP = re.compile(r"[*_`~>#|\[\]()!\\-]+")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class MarkdownBlock:
    lines: tuple[str, ...]
    kind: str  # "fence" | "heading" | "list" | "text"


@dataclass(frozen=True, slots=True)
class StrippedMarkdown:
    markdown: str
    removed_units: int
    removed_chars: int


def normalize_unit(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _LIST_ITEM.sub("", text, count=1)
    return _SPACES.sub(" ", _MARKUP.sub(" ", text)).strip()


def _digest(normalized: str) -> int:
    return int.from_bytes(hashlib.blake2b(normalized.encode(), digest_size=8).digest(), "big")


def split_blocks(markdown: str) -> list[MarkdownBlock]:
    """Blank-line separated blocks; a fenced code block is one block, blank lines included."""
    blocks: list[MarkdownBlock] = []
    current: list[str] = []
    fence: str | None = None

    def flush() -> None:
        if current:
            first = current[0]
            if _FENCE.match(first):
                kind = "fence"
            elif len(current) == 1 and _HEADING.match(first):
                kind = "heading"
            elif _LIST_ITEM.match(first):
                kind = "list"
            else:
                kind = "text"
            blocks.append(MarkdownBlock(tuple(current), kind))
            current.clear()

    for line in markdown.replace("\r\n", "\n").split("\n"):
        match = _FENCE.match(line)
        if fence is not None:
            current.append(line)
            if match and match.group(1) == fence:
                fence = None
                flush()
            continue
        if match:
            flush()
            fence = match.group(1)
            current.append(line)
        elif _HEADING.match(line):
            flush()
            current.append(line)
            flush()
        elif not line.strip():
            flush()
        else:
            current.append(line)
    flush()  # also closes an unterminated fence
    return blocks


def _list_items(block: MarkdownBlock) -> list[list[str]]:
    """A list block's items; indented continuation lines belong to the item above."""
    items: list[list[str]] = []
    for line in block.lines:
        if _LIST_ITEM.match(line) or not items:
            items.append([line])
        else:
            items[-1].append(line)
    return items


class BoilerplateIndex:
    def __init__(
        self,
        *,
        min_pages: int | None = None,
        min_ratio: float | None = None,
        keep_one_min_chars: int | None = None,
        config: Settings = settings,
    ) -> None:
        self.min_pages = min_pages if min_pages is not None else config.BOILERPLATE_MIN_PAGES
        self.min_ratio = min_ratio if min_ratio is not None else config.BOILERPLATE_MIN_RATIO
        self.keep_one_min_chars = (
            keep_one_min_chars
            if keep_one_min_chars is not None
            else config.BOILERPLATE_KEEP_ONE_MIN_CHARS
        )
        if self.min_pages < 2 or not 0 < self.min_ratio <= 1 or self.keep_one_min_chars < 0:
            raise ValueError("need min_pages >= 2, 0 < min_ratio <= 1, keep_one_min_chars >= 0")
        self._counts: Counter[int] = Counter()
        self._keeper: dict[int, str] = {}  # unit hash → first page key that had it
        self._page_keys: set[str] = set()
        self._contents: set[int] = set()

    @property
    def pages(self) -> int:
        """Distinct page contents counted so far."""
        return len(self._contents)

    def add(self, page_key: str, markdown: str) -> bool:
        """Count a page's units. False (not counted) for a known key or duplicate content."""
        if page_key in self._page_keys:
            return False
        self._page_keys.add(page_key)
        content = _digest(normalize_unit(markdown))
        if content in self._contents:
            return False
        self._contents.add(content)
        for digest in {d for _, d, _ in self._units(markdown)}:
            self._counts[digest] += 1
            self._keeper.setdefault(digest, page_key)
        return True

    def is_active(self) -> bool:
        return self.pages >= self.min_pages

    def _is_boilerplate(self, digest: int) -> bool:
        count = self._counts.get(digest, 0)
        return count >= 2 and count >= self.min_ratio * self.pages

    def boilerplate_count(self) -> int:
        if not self.is_active():
            return 0
        return sum(1 for digest in self._counts if self._is_boilerplate(digest))

    def strip(self, page_key: str, markdown: str) -> StrippedMarkdown:
        """`markdown` without its boilerplate units (unchanged while the index is inactive)."""
        if not self.is_active():
            return StrippedMarkdown(markdown, 0, 0)
        removed = removed_chars = 0
        out: list[str] = []
        for block in split_blocks(markdown):
            if block.kind in ("fence", "heading"):
                out.append("\n".join(block.lines))
                continue
            parts = _list_items(block) if block.kind == "list" else [list(block.lines)]
            kept: list[str] = []
            for part in parts:
                text = "\n".join(part)
                normalized = normalize_unit(text)
                if normalized and self._drop(page_key, _digest(normalized), normalized):
                    removed += 1
                    removed_chars += len(text)
                else:
                    kept.append(text)
            if kept:
                out.append("\n".join(kept))
        result = "\n\n".join(out).strip()
        return StrippedMarkdown(result + "\n" if result else "", removed, removed_chars)

    def _drop(self, page_key: str, digest: int, normalized: str) -> bool:
        if not self._is_boilerplate(digest):
            return False
        keep_here = (
            len(normalized) >= self.keep_one_min_chars and self._keeper.get(digest) == page_key
        )
        return not keep_here

    @staticmethod
    def _units(markdown: str) -> list[tuple[str, int, str]]:
        units: list[tuple[str, int, str]] = []
        for block in split_blocks(markdown):
            if block.kind in ("fence", "heading"):
                continue
            parts = _list_items(block) if block.kind == "list" else [list(block.lines)]
            for part in parts:
                normalized = normalize_unit("\n".join(part))
                if normalized:
                    units.append((block.kind, _digest(normalized), normalized))
        return units
