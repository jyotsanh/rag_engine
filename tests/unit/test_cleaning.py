"""Text cleaning: NFKC, invisible/control chars, whitespace, empty sections, min content."""

import pytest

from rag_engine.config import Settings
from rag_engine.crawling.fetchers.content import decode_html
from rag_engine.domain import PageReason, PageStatus
from rag_engine.extraction import (
    BoilerplateIndex,
    EmptyContentError,
    ExtractionInput,
    TextCleaner,
    default_registry,
    normalize_characters,
    page_marker,
)
from tests.conftest import OverrideSettings
from tests.support.fixture_site import SITE_ROOT

_BODY = "The WX-2048 sorter handles 1200 items per minute."  # enough content on its own


def _clean(markdown: str, *, min_content_chars: int = 20) -> str:
    return TextCleaner(min_content_chars=min_content_chars).clean(markdown).markdown


# ---- #
# Tests
# ---- #


class TestTextCleaning_S45:
    """S45 — Text cleaning: NFKC, zero-width/control chars, whitespace, pruning, min content."""

    # ---- characters ---- #

    @pytest.mark.parametrize(
        ("raw", "clean"),
        [
            ("\ufb01ne \ufb02ow", "fine flow"),  # ligatures
            ("\uff37\uff38\uff0d\uff12\uff10\uff14\uff18", "WX-2048"),  # full-width
            ("a\u00a0b\u2009c\u3000d", "a b c d"),  # NBSP, thin space, ideographic space
            ("zero\u200bwidth\u2060joined\ufeff", "zerowidthjoined"),
            ("soft\u00adhyphen", "softhyphen"),
            ("bidi\u202eevil\u202c \u2066iso\u2069 \u200emark\u200f", "bidievil iso mark"),
            ("bell\x07 null\x00 del\x7f c1\x85", "bell null del c1"),
        ],
        ids=["ligature", "fullwidth", "spaces", "zero-width", "soft-hyphen", "bidi", "controls"],
    )
    def test_normalize_characters(self, raw: str, clean: str) -> None:
        assert normalize_characters(raw) == clean

    def test_meaningful_joiners_are_kept(self) -> None:
        persian = "\u0645\u06cc\u200c\u062e\u0648\u0627\u0647\u0645"  # ZWNJ inside a word
        family = "\U0001f468\u200d\U0001f469\u200d\U0001f467"  # ZWJ emoji sequence

        assert normalize_characters(persian) == persian
        assert normalize_characters(family) == family

    def test_code_keeps_its_characters_but_loses_controls(self) -> None:
        line = "x\t=\u00a0\ufb01(1)\x07"

        assert normalize_characters(line, code=True) == "x\t=\u00a0\ufb01(1)"

    # ---- whitespace & code ---- #

    def test_whitespace_outside_code(self) -> None:
        markdown = (
            f"{_BODY}\r\nSecond   line\twith    gaps.   \r\n\n\n\n"
            "- item  one\n    - nested   item\n\n\n"
        )

        assert _clean(markdown) == (
            f"{_BODY}\nSecond line with gaps.\n\n- item one\n    - nested item\n"
        )

    def test_code_blocks_are_untouched(self) -> None:
        code = "```python\ndef f():\n\n\n\n    return  1\t# keep\u00a0this\n```"

        assert code in _clean(f"{_BODY}\n\n{code}\n")

    def test_unterminated_fence_protects_the_rest(self) -> None:
        markdown = f"{_BODY}\n\n```\nraw   spacing\n\n\n\nkept"

        assert _clean(markdown).endswith("```\nraw   spacing\n\n\n\nkept\n")

    # ---- structure ---- #

    def test_empty_list_items_and_blank_table_rows(self) -> None:
        markdown = (
            f"{_BODY}\n\n- first\n-\n- \n2.\n- last\n\n"
            "| a | b |\n|---|:---:|\n|  |  |\n| 1 | 2 |\n|   |   |\n"
        )

        assert _clean(markdown) == (
            f"{_BODY}\n\n- first\n- last\n\n| a | b |\n|---|:---:|\n| 1 | 2 |\n"
        )

    def test_empty_sections_are_pruned(self) -> None:
        markdown = (
            "# Guide\n\n"  # kept: its subsections have content
            "## Empty\n\n"  # pruned
            "## Install\n\n"  # kept: has a child with content
            "### Linux\n\n"
            f"{_BODY}\n\n"
            "### Windows\n\n"  # pruned: next heading is a sibling
            "## Links\n\n"  # pruned: only an empty child follows
            "### Nothing here\n"  # pruned
        )

        result = TextCleaner(min_content_chars=0).clean(markdown)

        assert result.markdown == f"# Guide\n\n## Install\n\n### Linux\n\n{_BODY}\n"
        assert result.pruned_headings == 4

    def test_heading_with_only_code_is_kept(self) -> None:
        markdown = f"# Intro\n\n{_BODY}\n\n## Example\n\n```bash\nwidgetctl list\n```\n"

        assert "## Example" in _clean(markdown)

    def test_empty_location_markers_are_pruned(self) -> None:
        markdown = (
            f"{page_marker(1)}\n\n# Spec\n\n{_BODY}\n\n"
            f"{page_marker(2)}\n\n"  # a blank page
            f"{page_marker(3)}\n\nPage three text.\n\n"
            f"{page_marker(4)}\n"  # trailing blank page
        )

        assert _clean(markdown) == (
            f"{page_marker(1)}\n\n# Spec\n\n{_BODY}\n\n{page_marker(3)}\n\nPage three text.\n"
        )

    def test_heading_continued_on_the_next_page_keeps_its_page(self) -> None:
        markdown = (
            f"{page_marker(1)}\n\n{_BODY}\n\n"
            f"{page_marker(2)}\n\n## Wiring\n\n"  # heading at the bottom of page 2
            f"{page_marker(3)}\n\nPin 1 carries power.\n"  # its text on page 3
        )

        assert _clean(markdown) == markdown

    # ---- min content ---- #

    @pytest.mark.parametrize(
        "markdown",
        [
            "",
            "\u200b\u200b\n\n   \n",
            "# Just a heading\n\n## And another\n",
            "Loading…",
            f"{page_marker(1)}\n\n---\n\n* * *\n",
        ],
        ids=["empty", "invisible-only", "headings-only", "loading", "markers-and-rules"],
    )
    def test_too_little_content_is_an_empty_content_skip(self, markdown: str) -> None:
        with pytest.raises(EmptyContentError) as exc:
            _clean(markdown)

        assert exc.value.detail == "below_min_content"
        assert (exc.value.reason, exc.value.page_status) == (
            PageReason.EMPTY_CONTENT,
            PageStatus.SKIPPED,
        )

    def test_content_chars_exclude_headings_and_markers(self) -> None:
        markdown = f"{page_marker(1)}\n\n# Long heading text here\n\nab 12\n"

        result = TextCleaner(min_content_chars=0).clean(markdown)

        assert result.content_chars == 4

    def test_threshold_comes_from_settings(self, override_settings: OverrideSettings) -> None:
        config: Settings = override_settings(CLEAN_MIN_CONTENT_CHARS=500)

        with pytest.raises(EmptyContentError):
            TextCleaner(config=config).clean(_BODY)

    # ---- idempotence & fixtures ---- #

    def test_cleaning_is_idempotent(self) -> None:
        messy = (
            "# T\n\n## Empty\n\nSome\u00a0text\u200b  here,\tand \ufb01ne.   \n\n\n- \n- item\n\n"
            "```\na  b\n\n\nc\n```\n\n| x |\n|---|\n|  |\n| 1 |\n"
        )

        once = _clean(messy)

        assert _clean(once) == once

    async def test_every_fixture_document_survives_cleaning(self) -> None:
        registry = default_registry()
        documents = [
            *(p for p in SITE_ROOT.rglob("*.html") if "app" not in p.parts),
            *(
                SITE_ROOT / "files" / name
                for name in (
                    "widget-spec.pdf",
                    "install-guide.docx",
                    "roadmap.pptx",
                    "pricing.xlsx",
                )
            ),
        ]
        cleaner = TextCleaner()

        for path in documents:
            body = path.read_bytes()
            text = decode_html(body, None)[0] if path.suffix == ".html" else None
            source = ExtractionInput(
                url=f"https://acme.example/{path.name}", mime_type="", body=body, text=text
            )
            if path.suffix == ".html":
                source = ExtractionInput(source.url, "text/html", body, text)
            markdown = (await registry.extract(source)).markdown

            cleaned = cleaner.clean(markdown).markdown

            assert cleaned == cleaner.clean(cleaned).markdown, path.name
            assert "\u200b" not in cleaned and "\x00" not in cleaned, path.name

    async def test_ingest_order_boilerplate_then_cleaning(self) -> None:
        # Step 40 → 45 at ingest, on the whole fixture site: the cookie banner is stripped,
        # then the home page's link-list headings (left empty by trafilatura) are pruned
        registry = default_registry()
        pages: dict[str, str] = {}
        for path in SITE_ROOT.rglob("*.html"):
            if "app" in path.parts:
                continue  # the unrendered SPA shell has no content
            body = path.read_bytes()
            source = ExtractionInput(
                f"https://acme.example/{path.relative_to(SITE_ROOT)}",
                "text/html",
                body,
                decode_html(body, None)[0],
            )
            pages[str(path.relative_to(SITE_ROOT))] = (await registry.extract(source)).markdown
        index = BoilerplateIndex(min_pages=5)
        for key, markdown in pages.items():
            index.add(key, markdown)

        home = pages["index.html"]
        cleaned = TextCleaner().clean(index.strip("index.html", home).markdown).markdown

        assert "## Start here" in home and "We use cookies" in home  # before
        assert "## Start here" not in cleaned and "We use cookies" not in cleaned
        assert "Acme Widgets builds modular widget assemblies" in cleaned
