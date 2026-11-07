"""Cross-page boilerplate: per-job unit frequency, stripped from Markdown."""

import pytest

from rag_engine.config import Settings
from rag_engine.crawling.fetchers.content import decode_html
from rag_engine.extraction import (
    BoilerplateIndex,
    EmptyContentError,
    ExtractionInput,
    HtmlExtractor,
    normalize_unit,
    split_blocks,
)
from tests.conftest import OverrideSettings
from tests.support.fixture_site import SITE_ROOT

_BANNER = "We use cookies to improve your experience. By continuing you accept our policy."
_NAV = "- [Home](/)\n- [Docs](/docs/)\n- [Blog](/blog/)"
_WARRANTY = (
    "Every Acme widget carries a two-year warranty covering parts and labour. Claims are "
    "handled by the regional service centre within ten working days of receipt, and a "
    "replacement unit ships the same day when the fault is confirmed."
)


def _page(n: int, *extra: str) -> str:
    return "\n\n".join(
        [
            f"# Page {n}",
            _BANNER,
            _NAV,
            f"Unique content number {n}: the WX-{n:04d} sorter handles {n * 100} items.",
            *extra,
        ]
    )


def _index(
    *, min_pages: int = 5, min_ratio: float = 0.5, keep_one_min_chars: int = 200
) -> BoilerplateIndex:
    return BoilerplateIndex(
        min_pages=min_pages, min_ratio=min_ratio, keep_one_min_chars=keep_one_min_chars
    )


def _site(index: BoilerplateIndex, pages: int, *extra: str) -> dict[str, str]:
    docs = {f"p{n}": _page(n, *extra) for n in range(pages)}
    for key, markdown in docs.items():
        index.add(key, markdown)
    return docs


# ---- #
# Tests
# ---- #


class TestBoilerplate_S40:
    """S40 — Cross-page boilerplate removal: per-job block-frequency index on Markdown."""

    # ---- blocks & normalization ---- #

    def test_split_blocks(self) -> None:
        markdown = (
            "# Title\nIntro line one\nline two\n\n- a\n- b\n  continued\n\n"
            "```python\nx = 1\n\ny = 2\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n## Next"
        )

        blocks = [(b.kind, len(b.lines)) for b in split_blocks(markdown)]

        assert blocks == [
            ("heading", 1),
            ("text", 2),
            ("list", 3),
            ("fence", 5),  # the blank line inside the fence stays in the block
            ("text", 3),  # table
            ("heading", 1),
        ]

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("We use **cookies**  here.", "we use cookies here."),
            ("- [Home](/)", "* [Home](/)"),
            ("1. First step", "2) First step"),
            ("\uff26\uff55\uff4c\uff4c width", "full width"),  # full-width letters: NFKC
        ],
    )
    def test_normalization_matches_variants(self, a: str, b: str) -> None:
        assert normalize_unit(a) == normalize_unit(b)

    # ---- stripping ---- #

    def test_frequent_units_are_stripped(self) -> None:
        index = _index()
        docs = _site(index, 6)

        result = index.strip("p3", docs["p3"])

        assert _BANNER not in result.markdown
        assert "[Home]" not in result.markdown
        assert "Unique content number 3" in result.markdown
        assert result.markdown.startswith("# Page 3\n")
        assert result.removed_units == 4  # banner + three nav items
        assert result.removed_chars > len(_BANNER)

    def test_nothing_is_stripped_below_min_pages(self) -> None:
        index = _index(min_pages=5)
        docs = _site(index, 4)

        result = index.strip("p0", docs["p0"])

        assert not index.is_active()
        assert result.markdown == docs["p0"]
        assert result.removed_units == 0

    def test_units_below_the_ratio_are_kept(self) -> None:
        shared_by_few = "Seasonal notice: the Springfield office closes in August."
        docs = {f"p{n}": _page(n, *([shared_by_few] if n < 4 else [])) for n in range(10)}
        index = _index(min_ratio=0.5)
        for key, markdown in docs.items():
            index.add(key, markdown)

        markdown = index.strip("p0", docs["p0"]).markdown

        assert shared_by_few in markdown  # on 4 of 10 pages: below the 50 % ratio
        assert _BANNER not in markdown  # on 10 of 10

    def test_nav_items_go_even_when_the_active_item_differs(self) -> None:
        index = _index()
        docs = {
            f"p{n}": (
                f"# P{n}\n\n- [Home](/)\n- [Docs](/docs/)\n- **Page {n}** (you are here)"
                f"\n\nBody {n}."
            )
            for n in range(6)
        }
        for key, markdown in docs.items():
            index.add(key, markdown)

        markdown = index.strip("p2", docs["p2"]).markdown

        assert "[Home]" not in markdown and "[Docs]" not in markdown
        assert "- **Page 2** (you are here)" in markdown  # the varying item survives

    def test_headings_and_code_are_never_stripped(self) -> None:
        shared_code = "```bash\npip install acme-widgets\n```"
        index = _index()
        docs = _site(index, 6, "## Installation", shared_code)

        markdown = index.strip("p4", docs["p4"]).markdown

        assert "## Installation" in markdown
        assert shared_code in markdown

    def test_fenced_code_is_reassembled_byte_for_byte(self) -> None:
        code = "```python\ndef f():\n\n    return 1\n```"
        index = _index()
        docs = _site(index, 6, code)

        assert code in index.strip("p1", docs["p1"]).markdown

    # ---- duplicates & keeper ---- #

    def test_duplicate_pages_count_once(self) -> None:
        index = _index()
        shared = _page(99)
        for n in range(10):  # ten URLs serving the same page
            index.add(f"dup{n}", shared)
        for n in range(4):
            index.add(f"p{n}", _page(n))

        result = index.strip("dup0", shared)

        assert index.pages == 5  # 1 distinct duplicate + 4 others
        assert "Unique content number 99" in result.markdown  # not "on 10 of 14 pages"

    def test_add_is_idempotent_per_page_key(self) -> None:
        index = _index()

        assert index.add("p0", _page(0)) is True
        assert index.add("p0", _page(1)) is False
        assert index.pages == 1

    def test_long_shared_block_keeps_one_copy(self) -> None:
        index = _index(keep_one_min_chars=200)
        docs = _site(index, 6, _WARRANTY)

        on_keeper = index.strip("p0", docs["p0"]).markdown  # p0 was added first
        elsewhere = index.strip("p3", docs["p3"]).markdown

        assert _WARRANTY in on_keeper
        assert _WARRANTY not in elsewhere
        assert _BANNER not in on_keeper  # short chrome goes everywhere, keeper included

    def test_keep_one_can_be_disabled(self) -> None:
        index = _index(keep_one_min_chars=10_000)
        docs = _site(index, 6, _WARRANTY)

        assert _WARRANTY not in index.strip("p0", docs["p0"]).markdown

    def test_pages_added_later_are_stripped_too(self) -> None:
        index = _index()
        _site(index, 6)

        result = index.strip("late", _page(42))

        assert _BANNER not in result.markdown
        assert "Unique content number 42" in result.markdown

    def test_page_made_only_of_boilerplate_becomes_empty(self) -> None:
        index = _index()
        _site(index, 6)

        result = index.strip("x", f"{_BANNER}\n\n{_NAV}")

        assert result.markdown == ""  # Step 45's min-content check turns this into a skip

    @pytest.mark.parametrize(
        "kwargs",
        [{"min_pages": 1}, {"min_ratio": 0}, {"min_ratio": 1.5}, {"keep_one_min_chars": -1}],
    )
    def test_invalid_thresholds(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            _index(**kwargs)

    def test_defaults_come_from_settings(self, override_settings: OverrideSettings) -> None:
        config: Settings = override_settings(
            BOILERPLATE_MIN_PAGES=3, BOILERPLATE_MIN_RATIO=0.8, BOILERPLATE_KEEP_ONE_MIN_CHARS=50
        )

        index = BoilerplateIndex(config=config)

        assert (index.min_pages, index.min_ratio, index.keep_one_min_chars) == (3, 0.8, 50)

    # ---- fixture site ---- #

    async def test_fixture_cookie_banner_is_removed(self) -> None:
        extractor = HtmlExtractor()
        docs: dict[str, str] = {}
        for path in sorted(SITE_ROOT.rglob("*.html")):
            body = path.read_bytes()
            text, encoding = decode_html(body, None)
            source = ExtractionInput(
                url=f"https://acme.example/{path.relative_to(SITE_ROOT)}",
                mime_type="text/html",
                body=body,
                text=text,
                encoding=encoding,
            )
            try:
                docs[str(path.relative_to(SITE_ROOT))] = (await extractor.extract(source)).markdown
            except EmptyContentError:
                continue  # the unrendered SPA shell
        index = BoilerplateIndex(config=Settings(_env_file=None, BOILERPLATE_MIN_PAGES=5))
        for key, markdown in docs.items():
            index.add(key, markdown)
        banner = "We use cookies to improve your experience on Acme Widgets."

        stripped = {key: index.strip(key, markdown).markdown for key, markdown in docs.items()}

        assert index.pages == len(docs) - 1  # dup-a and dup-b serve the same content
        assert sum(banner in markdown for markdown in docs.values()) >= 5  # it was there
        assert not any(banner in markdown for markdown in stripped.values())
        assert "Crème brûlée" in stripped["latin1.html"]
        guide = stripped["docs/getting-started.html"]
        assert "```bash\ncurl -fsSL https://acme.example/install.sh | sh" in guide
        assert "### Verify the registration" in guide
