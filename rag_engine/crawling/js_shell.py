"""Does this HTML need a browser? Heuristics for client-rendered "JS shell" pages.

A score over cheap signals from the raw HTML; at `threshold` or more the page is re-rendered
with Playwright (Step 36). Rendering is expensive, so the detector leans towards "no":

| signal                | weight | meaning                                                     |
|-----------------------|--------|-------------------------------------------------------------|
| very_low_text         | +3     | < 50 visible characters                                     |
| low_text              | +2     | < 250 visible characters                                    |
| empty_mount           | +3     | SPA mount node (`#root`, `#__next`, `<app-root>`) has no text |
| noscript_hint         | +2     | `<noscript>` says JavaScript is required                    |
| template_markers      | +2     | ≥ 3 unrendered `{{ … }}` placeholders in the visible text    |
| ajax_fragment_meta    | +2     | `<meta name="fragment" content="!">` (AJAX crawling scheme)  |
| script_heavy          | +1     | script weight >= 10 x the visible text                       |
| rich_text             | -2     | ≥ 2000 visible characters: the content is already there      |

A page without any executable `<script>` never needs rendering, whatever the score.
Visible text excludes `<script>`, `<style>`, `<noscript>`, `<template>`, `<svg>` and `<head>`.
JSON data blocks (`application/ld+json`, `__NEXT_DATA__`) are data, not code, so they don't count.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

import lxml.html
from lxml import etree

_MOUNT_IDS = frozenset(
    {
        "root",
        "app",
        "__next",
        "__nuxt",
        "___gatsby",
        "svelte",
        "app-root",
        "main-app",
        "application",
    }
)
_MOUNT_TAGS = frozenset({"app-root"})  # Angular's default root component
_MOUNT_ATTRS = ("ng-app", "data-reactroot", "data-v-app")
_NON_VISIBLE = ("script", "style", "noscript", "template", "svg", "head")
_DATA_SCRIPT_TYPES = frozenset(
    {"application/ld+json", "application/json", "text/template", "text/x-template"}
)
_NOSCRIPT_HINT = re.compile(
    r"(enable|turn on|activate|requires?|need|needs|is required|disabled|support)"
    r"[^.]{0,40}javascript|javascript[^.]{0,40}(enable|required|disabled|turned off|needed)",
    re.I,
)
_TEMPLATE_MARKER = re.compile(r"\{\{[^{}]{1,80}\}\}")
_WHITESPACE = re.compile(r"\s+")
_EXTERNAL_SCRIPT_WEIGHT = 5_000  # an external bundle's size is unknown; assume a typical chunk
_NOT_HIDDEN = "[not(" + " or ".join(f"ancestor::{tag}" for tag in _NON_VISIBLE) + ")]"
_VISIBLE_TEXT = etree.XPath("//body//text()" + _NOT_HIDDEN)
_VISIBLE_TEXT_BELOW = etree.XPath(".//text()" + _NOT_HIDDEN)


class JsSignal(StrEnum):
    VERY_LOW_TEXT = "very_low_text"
    LOW_TEXT = "low_text"
    EMPTY_MOUNT = "empty_mount"
    NOSCRIPT_HINT = "noscript_hint"
    TEMPLATE_MARKERS = "template_markers"
    AJAX_FRAGMENT_META = "ajax_fragment_meta"
    SCRIPT_HEAVY = "script_heavy"
    RICH_TEXT = "rich_text"


_WEIGHTS: dict[JsSignal, int] = {
    JsSignal.VERY_LOW_TEXT: 3,
    JsSignal.LOW_TEXT: 2,
    JsSignal.EMPTY_MOUNT: 3,
    JsSignal.NOSCRIPT_HINT: 2,
    JsSignal.TEMPLATE_MARKERS: 2,
    JsSignal.AJAX_FRAGMENT_META: 2,
    JsSignal.SCRIPT_HEAVY: 1,
    JsSignal.RICH_TEXT: -2,
}


@dataclass(frozen=True, slots=True)
class JsShellThresholds:
    threshold: int = 3
    very_low_text: int = 50
    low_text: int = 250
    rich_text: int = 2_000
    script_ratio: float = 10.0
    template_markers: int = 3


@dataclass(frozen=True, slots=True)
class JsShellVerdict:
    needs_rendering: bool
    score: int
    signals: tuple[JsSignal, ...]
    text_chars: int  # visible text, whitespace-collapsed
    script_count: int  # executable scripts, inline and external
    script_bytes: int  # inline code only
    mount: str | None = None  # the empty mount node, e.g. "#root" or "<app-root>"


class JsShellDetector:
    def __init__(self, thresholds: JsShellThresholds | None = None) -> None:
        self._t = thresholds or JsShellThresholds()

    def detect(self, html: str) -> JsShellVerdict:
        root = _parse(html)
        if root is None:
            return JsShellVerdict(False, 0, (), 0, 0, 0)

        text = _visible_text(_VISIBLE_TEXT, root)
        script_count, script_bytes = _scripts(root)
        mount = _empty_mount(root)

        signals: list[JsSignal] = []
        if len(text) < self._t.very_low_text:
            signals.append(JsSignal.VERY_LOW_TEXT)
        elif len(text) < self._t.low_text:
            signals.append(JsSignal.LOW_TEXT)
        elif len(text) >= self._t.rich_text:
            signals.append(JsSignal.RICH_TEXT)
        if mount is not None:
            signals.append(JsSignal.EMPTY_MOUNT)
        if any(_NOSCRIPT_HINT.search(_text_of(n)) for n in root.iter("noscript")):
            signals.append(JsSignal.NOSCRIPT_HINT)
        if len(_TEMPLATE_MARKER.findall(text)) >= self._t.template_markers:
            signals.append(JsSignal.TEMPLATE_MARKERS)
        if any(
            (m.get("name") or "").lower() == "fragment" and (m.get("content") or "").strip() == "!"
            for m in root.iter("meta")
        ):
            signals.append(JsSignal.AJAX_FRAGMENT_META)
        weight = script_bytes + _EXTERNAL_SCRIPT_WEIGHT * _external_count(root)
        if script_count and weight >= self._t.script_ratio * max(len(text), 1):
            signals.append(JsSignal.SCRIPT_HEAVY)

        score = sum(_WEIGHTS[s] for s in signals)
        # nothing executable: there is nothing a browser could add
        needs = script_count > 0 and score >= self._t.threshold
        return JsShellVerdict(
            needs, score, tuple(signals), len(text), script_count, script_bytes, mount
        )


def _parse(html: str) -> lxml.html.HtmlElement | None:
    if not html or not html.strip():
        return None
    parser = lxml.html.HTMLParser(encoding="utf-8")
    try:
        return lxml.html.document_fromstring(html.encode("utf-8", errors="replace"), parser=parser)
    except etree.ParserError, ValueError:
        return None


def _visible_text(xpath: etree.XPath, element: lxml.html.HtmlElement) -> str:
    result = xpath(element)
    parts = [str(t) for t in result] if isinstance(result, list) else []
    return _WHITESPACE.sub(" ", " ".join(parts)).strip()


def _text_of(element: lxml.html.HtmlElement) -> str:
    return _WHITESPACE.sub(" ", element.text_content()).strip()


def _is_code(script: lxml.html.HtmlElement) -> bool:
    kind = (script.get("type") or "").strip().lower()
    return kind not in _DATA_SCRIPT_TYPES and script.get("id") != "__NEXT_DATA__"


def _scripts(root: lxml.html.HtmlElement) -> tuple[int, int]:
    count = inline_bytes = 0
    for script in root.iter("script"):
        if not _is_code(script):
            continue
        count += 1
        if not script.get("src"):
            inline_bytes += len((script.text or "").strip())
    return count, inline_bytes


def _external_count(root: lxml.html.HtmlElement) -> int:
    return sum(1 for s in root.iter("script") if s.get("src") and _is_code(s))


def _empty_mount(root: lxml.html.HtmlElement) -> str | None:
    """The first SPA mount node whose subtree has no visible text (spinners don't count)."""
    for element in root.iter():
        if not isinstance(element.tag, str):  # comments, processing instructions
            continue
        label: str | None = None
        element_id = (element.get("id") or "").strip()
        if element_id.lower() in _MOUNT_IDS:
            label = f"#{element_id}"
        elif element.tag.lower() in _MOUNT_TAGS:
            label = f"<{element.tag.lower()}>"
        elif any(element.get(attr) is not None for attr in _MOUNT_ATTRS):
            label = next(f"[{a}]" for a in _MOUNT_ATTRS if element.get(a) is not None)
        if label is None or element.tag.lower() in ("html", "body"):
            continue
        if not _visible_text(_VISIBLE_TEXT_BELOW, element):  # a spinner <div> is still empty
            return label
    return None
