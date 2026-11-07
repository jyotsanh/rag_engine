"""Contextual embed text: what a chunk means depends on where it sits (Step 58).

A chunk like "Run the installer and restart." embeds poorly on its own: which product,
which page, which section? So the embedder gets a short **context** in front of the text:

    Install guide | Acme Widgets          <- the page title
    Getting started > Linux               <- the heading path (outermost first)

    Run the installer and restart.        <- the chunk text (`Chunk.text`, shown to users)

Rules:
- A leading heading that only repeats the title is dropped: title "Install guide | Acme"
  with `# Install guide` gives no second "Install guide" line (comparison is
  case/space-insensitive; the title may extend the heading after `|`, `:`, a middle dot, or
  a spaced hyphen/en/em dash; a hyphen inside a word like "WX-2048" is not a separator).
- No title and no headings: the embed text is the chunk text itself.
- The context is **bounded** by `context_tokens` (`CHUNK_CONTEXT_MAX_TOKENS`), so the
  splitter's reservation (Step 57) is predictable and the chunk text always gets most of
  the budget. Over the budget: middle headings collapse to "…" (the outermost and the
  innermost are what disambiguate), then the title is shortened, then the remaining
  headings; shortened parts end in "…". A budget of 0 disables the context.
"""

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # base.py imports this module
    from rag_engine.chunking.base import Tokenizer

ELLIPSIS = "…"
PATH_SEPARATOR = " > "
# "|", ":", "·" anywhere; hyphen / en dash / em dash only with spaces around them
_TITLE_SEPARATORS = re.compile(r"\s+[-\u2013\u2014]\s+|\s*[|:\u00b7]\s*")


def _normal(text: str) -> str:
    return " ".join(text.split()).casefold()


def _repeats_title(heading: str, title: str) -> bool:
    heading, title = _normal(heading), _normal(title)
    if not heading or not title:
        return False
    if heading == title:
        return True
    return any(_normal(part) == heading for part in _TITLE_SEPARATORS.split(title))


def _shorten(text: str, max_tokens: int, tokenizer: Tokenizer) -> str:
    """The longest word prefix of `text` (plus "…") within `max_tokens`; "" if none fits."""
    if tokenizer.count(text) <= max_tokens:
        return text
    words = text.split()
    low, high = 0, len(words)
    while low < high:  # most words such that "words…" fits
        middle = (low + high + 1) // 2
        if tokenizer.count(" ".join(words[:middle]) + ELLIPSIS) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    if low:
        return " ".join(words[:low]) + ELLIPSIS
    # a single very long first word: cut by characters
    word = words[0] if words else text
    low, high = 0, len(word)
    while low < high:
        middle = (low + high + 1) // 2
        if tokenizer.count(word[:middle] + ELLIPSIS) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return word[:low] + ELLIPSIS if low else ""


def build_context(
    title: str | None,
    heading_path: Sequence[str],
    *,
    tokenizer: Tokenizer,
    max_tokens: int,
) -> str:
    """The context lines for a chunk (possibly ""), within `max_tokens`."""
    if max_tokens <= 0:
        return ""
    title = " ".join((title or "").split())
    path = [" ".join(h.split()) for h in heading_path if h and h.strip()]
    if title and path and _repeats_title(path[0], title):
        path = path[1:]

    def render(title_part: str, headings: Sequence[str]) -> str:
        lines = [title_part] if title_part else []
        if headings:
            lines.append(PATH_SEPARATOR.join(headings))
        return "\n".join(lines)

    def fits(text: str) -> bool:
        return tokenizer.count(text) <= max_tokens

    context = render(title, path)
    if fits(context):
        return context
    # 1. collapse the middle of a deep path, keeping the outermost and innermost headings
    if len(path) > 2:
        path = [path[0], ELLIPSIS, path[-1]]
        context = render(title, path)
        if fits(context):
            return context
    # 2. shorten the title to what the path leaves (at least a third of the budget)
    path_tokens = tokenizer.count(PATH_SEPARATOR.join(path)) if path else 0
    title_budget = max(max_tokens - path_tokens - 1, max_tokens // 3)
    title = _shorten(title, title_budget, tokenizer) if title else ""
    context = render(title, path)
    if fits(context):
        return context
    # 3. shorten the headings, innermost kept longest: share what the title left
    left = max_tokens - (tokenizer.count(title) + 1 if title else 0)
    if path and left > 0:
        share = max(left // len(path) - 1, 1)
        path = [h if h == ELLIPSIS else _shorten(h, share, tokenizer) for h in path]
        path = [h for h in path if h]
        context = render(title, path)
        if fits(context):
            return context
    # 4. whatever still fits: the title alone, else nothing
    return title if title and fits(title) else ""


def compose_embed_text(context: str, text: str) -> str:
    return f"{context}\n\n{text}" if context else text
