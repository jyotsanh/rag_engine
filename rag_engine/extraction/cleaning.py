"""Text cleaning for extracted Markdown, before dedupe hashing and chunking.

Character level (code fences are left alone except for control characters):
- **NFKC** outside code: ligatures (`ﬁ` → `fi`), full-width forms, compatibility spaces
  (NBSP, thin space, ideographic space → space).
- **Invisible characters** removed: zero-width space, word joiner, BOM, soft hyphen,
  Mongolian vowel separator, bidi embeddings/overrides/isolates and LRM/RLM (also the
  "Trojan source" trick). **ZWJ/ZWNJ are kept**: they carry meaning in Persian, Indic
  scripts and emoji sequences.
- **Control characters** removed everywhere (C0 except tab/newline, DEL, C1); inside code
  tabs survive.
- **Whitespace** outside code: runs of spaces/tabs collapsed to one space *after* the
  leading indentation (which nests lists), trailing spaces stripped, blank runs → one
  blank line (code blocks keep theirs).

Structure:
- Empty list items (`- ` with nothing after) and all-blank table body rows are dropped.
- **Empty sections are pruned**: a heading is removed when nothing but headings and
  location markers follows it before the next heading of the same or a higher level. A
  parent heading whose subsections have content stays.
- Location markers (`<!-- page|slide|sheet N -->`) with no content before the next marker
  are removed too.

Finally a **min-content threshold**: fewer than `CLEAN_MIN_CONTENT_CHARS` letters/digits →
`EmptyContentError` (the page is skipped as `empty_content`).

`clean()` is idempotent, because it runs twice per page: during the crawl (empty pages are
skipped early, Step 46 hashes clean text) and after cross-page boilerplate stripping at
ingest (Step 40 can empty sections).
"""

import re
import unicodedata
from dataclasses import dataclass

from rag_engine.config import Settings, settings
from rag_engine.extraction.boilerplate import split_blocks
from rag_engine.extraction.exceptions import EmptyContentError

_FENCE = re.compile(r"^\s{0,3}(```|~~~)")
_HEADING = re.compile(r"^\s{0,3}(#{1,6})(\s|$)")
_MARKER = re.compile(r"^<!--\s*(page|slide|sheet)\s+\d+\s*-->$", re.I)
_COMMENT = re.compile(r"^<!--.*-->$", re.S)
_EMPTY_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d{1,9}[.)])\s*$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$")
_INNER_SPACES = re.compile(r"[ \t]+")

# invisible / formatting characters stripped everywhere (ZWJ U+200D and ZWNJ U+200C kept)
_INVISIBLE = dict.fromkeys(
    [
        0x00AD,  # soft hyphen
        0x180E,  # Mongolian vowel separator
        0x200B,  # zero-width space
        0x200E, 0x200F,  # LRM, RLM
        0x202A, 0x202B, 0x202C, 0x202D, 0x202E,  # bidi embeddings / overrides
        0x2060,  # word joiner
        0x2066, 0x2067, 0x2068, 0x2069,  # bidi isolates
        0xFEFF,  # BOM / zero-width no-break space
    ]
)  # fmt: skip
# control characters: C0 except \t and \n, DEL, C1
_CONTROLS = dict.fromkeys(
    [c for c in range(0x20) if c not in (0x09, 0x0A)] + [0x7F] + list(range(0x80, 0xA0))
)


@dataclass(frozen=True, slots=True)
class CleanedText:
    markdown: str
    content_chars: int  # letters and digits outside headings and markers
    pruned_headings: int


def normalize_characters(text: str, *, code: bool = False) -> str:
    """Character-level cleaning of one line. `code=True`: controls only (tabs kept)."""
    text = text.translate(_CONTROLS).translate(_INVISIBLE)
    if code:
        return text
    return unicodedata.normalize("NFKC", text).translate(_INVISIBLE)


class TextCleaner:
    def __init__(self, *, min_content_chars: int | None = None, config: Settings = settings):
        self.min_content_chars = (
            min_content_chars if min_content_chars is not None else config.CLEAN_MIN_CONTENT_CHARS
        )

    def clean(self, markdown: str) -> CleanedText:
        """Cleaned Markdown, or `EmptyContentError` below the min-content threshold."""
        lines = self._clean_lines(markdown)
        # blocks are rejoined with exactly one blank line; fenced blocks keep their own
        text, pruned = _prune("\n".join(lines).strip())
        content_chars = _content_chars(text)
        if content_chars < self.min_content_chars:
            raise EmptyContentError(
                f"only {content_chars} letters/digits of content "
                f"(minimum {self.min_content_chars})",
                detail="below_min_content",
            )
        return CleanedText(text + "\n", content_chars, pruned)

    @staticmethod
    def _clean_lines(markdown: str) -> list[str]:
        out: list[str] = []
        fence: str | None = None
        for raw in markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            match = _FENCE.match(raw)
            if fence is not None:  # inside a code block
                out.append(normalize_characters(raw, code=True).rstrip())
                if match and match.group(1) == fence:
                    fence = None
                continue
            if match:
                fence = match.group(1)
                out.append(normalize_characters(raw, code=True).strip())
                continue
            line = normalize_characters(raw).rstrip()
            indent = line[: len(line) - len(line.lstrip(" \t"))].replace("\t", "    ")
            body = _INNER_SPACES.sub(" ", line.lstrip(" \t"))
            if not body:
                out.append("")
            elif _EMPTY_LIST_ITEM.match(body) or _is_blank_table_row(body):
                continue
            else:
                out.append(indent + body)
        return out


def _is_blank_table_row(line: str) -> bool:
    if not _TABLE_ROW.match(line) or _TABLE_SEPARATOR.match(line):
        return False
    return not line.strip().strip("|").replace("|", "").strip()


def _block_kind(block_lines: tuple[str, ...], kind: str) -> tuple[str, int]:
    """("heading", level) | ("marker", 0) | ("content", 0)."""
    if kind == "heading":
        match = _HEADING.match(block_lines[0])
        return ("heading", len(match.group(1))) if match else ("content", 0)
    joined = "\n".join(block_lines).strip()
    if kind == "text" and (_MARKER.match(joined) or _COMMENT.match(joined)):
        return "marker", 0
    return "content", 0


def _prune(markdown: str) -> tuple[str, int]:
    blocks = split_blocks(markdown)
    kinds = [_block_kind(b.lines, b.kind) for b in blocks]
    keep = [True] * len(blocks)

    # 1. headings: a section may continue past a page marker (a heading at the bottom of a
    #    PDF page, its text on the next), so markers don't end a heading's section
    for i, (kind, level) in enumerate(kinds):
        if kind != "heading":
            continue
        has_content = False
        for later_kind, later_level in kinds[i + 1 :]:
            if later_kind == "heading" and later_level <= level:
                break
            if later_kind == "content":
                has_content = True
                break
        keep[i] = has_content

    # 2. markers: kept when anything kept (content or a kept heading) follows before the
    #    next marker, so a heading never appears to move to the previous page
    for i, (kind, _) in enumerate(kinds):
        if kind != "marker":
            continue
        keep[i] = False
        for j in range(i + 1, len(kinds)):
            if kinds[j][0] == "marker":
                break
            if keep[j]:
                keep[i] = True
                break

    pruned = sum(1 for (kind, _), k in zip(kinds, keep, strict=True) if kind == "heading" and not k)
    text = "\n\n".join("\n".join(b.lines) for b, k in zip(blocks, keep, strict=True) if k)
    return text.strip(), pruned


def _content_chars(markdown: str) -> int:
    total = 0
    for block in split_blocks(markdown):
        if _block_kind(block.lines, block.kind)[0] != "content":
            continue
        total += sum(1 for ch in "\n".join(block.lines) if ch.isalnum())
    return total
