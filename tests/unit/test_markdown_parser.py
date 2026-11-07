"""Markdown structure parser: sections by heading, atomic blocks, location markers."""

import pytest

from rag_engine.chunking import Block, BlockKind, ParsedDocument, parse_markdown
from rag_engine.extraction import ExtractionInput, default_registry
from tests.support.fixture_site import SITE_ROOT


def _kinds(doc: ParsedDocument) -> list[BlockKind]:
    return [block.kind for block in doc.blocks]


def _only(doc: ParsedDocument, kind: BlockKind) -> Block:
    (block,) = [b for b in doc.blocks if b.kind is kind]
    return block


def _fixture(name: str) -> bytes:
    return (SITE_ROOT / "files" / name).read_bytes()


async def _extracted(name: str, mime: str) -> str:
    body = _fixture(name)
    document = await default_registry().extract(
        ExtractionInput(url=f"https://acme.example/files/{name}", mime_type=mime, body=body)
    )
    return document.markdown


# ---- #
# Tests
# ---- #


class TestMarkdownParser_S56:
    """S56 — Markdown structure parser: heading hierarchy, atomic blocks, location markers."""

    # ---- sections ---- #

    def test_heading_hierarchy_builds_paths(self) -> None:
        doc = parse_markdown(
            "Intro text.\n\n"
            "# Guide\n\nAbout.\n\n"
            "## Install\n\nSteps.\n\n"
            "### Linux\n\napt.\n\n"
            "## Configure\n\nKeys.\n\n"
            "# Reference\n\nAPI.\n"
        )

        assert [(s.heading_path, s.level) for s in doc.sections] == [
            ((), 0),
            (("Guide",), 1),
            (("Guide", "Install"), 2),
            (("Guide", "Install", "Linux"), 3),
            (("Guide", "Configure"), 2),  # a sibling closes "Install > Linux"
            (("Reference",), 1),
        ]
        assert [b.text for b in doc.sections[0].blocks] == ["Intro text."]
        assert [b.text for b in doc.sections[3].blocks] == ["apt."]

    def test_skipped_levels_and_setext_headings(self) -> None:
        doc = parse_markdown("# A\n\n### Deep\n\nx\n\nB\n-\n\ny\n\nC\n===\n\nz\n")

        assert [s.heading_path for s in doc.sections] == [
            ("A",),
            ("A", "Deep"),
            ("A", "B"),  # setext h2 closes the h3
            ("C",),
        ]
        assert doc.sections[2].heading_markdown == "B\n-"

    def test_heading_text_is_plain(self) -> None:
        doc = parse_markdown(
            "## The **`run()`** call — see [docs](https://a.example) ![ic](i.png)\n"
        )

        (section,) = doc.sections
        assert section.heading == "The run() call — see docs ic"
        assert section.heading_markdown is not None and section.heading_markdown.startswith("## ")

    def test_empty_headings_are_kept_and_an_empty_preamble_dropped(self) -> None:
        doc = parse_markdown("# Only a title\n## And a subtitle\n")

        assert [(s.heading_path, s.blocks) for s in doc.sections] == [
            (("Only a title",), []),
            (("Only a title", "And a subtitle"), []),
        ]
        assert parse_markdown("").sections == ()
        assert parse_markdown("\n\n   \n").sections == ()

    def test_hashes_inside_code_and_tables_are_not_headings(self) -> None:
        doc = parse_markdown(
            "# Real\n\n```bash\n# not a heading\nls\n```\n\n| # | x |\n|---|---|\n| # | y |\n"
        )

        assert [s.heading for s in doc.sections] == ["Real"]
        assert _kinds(doc) == [BlockKind.CODE, BlockKind.TABLE]

    # ---- blocks ---- #

    def test_block_kinds_and_atomicity(self) -> None:
        doc = parse_markdown(
            "Para one\nsame para.\n\n> quoted\n> more\n\n- a\n- b\n\n1. x\n2. y\n\n"
            "| h |\n|---|\n| v |\n\n```\ncode\n```\n\n    indented\n\n<div>raw</div>\n"
        )

        assert _kinds(doc) == [
            BlockKind.PARAGRAPH,
            BlockKind.QUOTE,
            BlockKind.LIST,
            BlockKind.LIST,
            BlockKind.TABLE,
            BlockKind.CODE,
            BlockKind.CODE,
            BlockKind.HTML,
        ]
        assert [b.atomic for b in doc.blocks] == [False, False, True, True, True, True, True, False]
        assert doc.blocks[0].text == "Para one\nsame para."
        assert doc.blocks[1].text == "> quoted\n> more"

    def test_table_parts_are_rows_under_a_header(self) -> None:
        doc = parse_markdown(
            "| Model | Speed |\n| --- | ---: |\n| WX-2048 | 1200 |\n| WX-4096 \\| XL | 2400 |\n"
        )

        table = _only(doc, BlockKind.TABLE)
        assert table.header == "| Model | Speed |\n| --- | ---: |"
        assert table.parts == ("| WX-2048 | 1200 |", "| WX-4096 \\| XL | 2400 |")
        assert table.text == "\n".join([table.header, *table.parts])

    def test_fenced_code_keeps_fences_language_and_lines(self) -> None:
        doc = parse_markdown("~~~~python\ndef f():\n    return 1\n\n# end\n~~~~\n")

        code = _only(doc, BlockKind.CODE)
        assert (code.header, code.footer, code.info) == ("~~~~python", "~~~~", "python")
        assert code.parts == ("def f():", "    return 1", "", "# end")  # blank line kept

    def test_an_unclosed_fence_runs_to_the_end(self) -> None:
        doc = parse_markdown("Text.\n\n```js\nlet a = 1\nlet b = 2\n")

        code = _only(doc, BlockKind.CODE)
        assert (code.header, code.footer, code.info) == ("```js", None, "js")
        assert code.parts == ("let a = 1", "let b = 2")

    def test_indented_code_has_no_fences(self) -> None:
        doc = parse_markdown("Text.\n\n    line 1\n      line 2\n")

        code = _only(doc, BlockKind.CODE)
        assert (code.header, code.footer, code.info) == (None, None, None)
        assert code.parts == ("    line 1", "      line 2")

    def test_list_parts_are_top_level_items_with_their_children(self) -> None:
        doc = parse_markdown("- one\n  - nested a\n  - nested b\n\n- two\n  continued\n- three\n")

        items = _only(doc, BlockKind.LIST).parts
        assert items == ("- one\n  - nested a\n  - nested b", "- two\n  continued", "- three")

    # ---- what is not content ---- #

    def test_location_markers_set_locations_and_are_not_content(self) -> None:
        doc = parse_markdown(
            "<!-- page 1 -->\n# Spec\n\nFirst page.\n\n<!-- page 2 -->\nSecond page.\n\n"
            "## Interface\n\nStill two.\n<!-- page 3 -->\n| a |\n|---|\n| b |\n"
        )

        assert doc.location_kind == "page"
        assert [(s.heading, s.location) for s in doc.sections] == [
            ("Spec", 1),
            ("Interface", 2),
        ]
        assert [(b.text, b.location) for b in doc.blocks] == [
            ("First page.", 1),
            ("Second page.", 2),
            ("Still two.", 2),
            ("| a |\n|---|\n| b |", 3),
        ]
        assert not any("<!--" in b.text for b in doc.blocks)

    @pytest.mark.parametrize("kind", ["slide", "sheet", "PAGE"])
    def test_every_marker_kind_is_a_location(self, kind: str) -> None:
        doc = parse_markdown(f"<!-- {kind} 7 -->\nText.\n")

        assert doc.location_kind == kind.lower()
        assert [b.location for b in doc.blocks] == [7]

    def test_comments_and_thematic_breaks_are_dropped(self) -> None:
        doc = parse_markdown("A.\n\n<!-- generated by tool -->\n\n---\n\n***\n\nB.\n")

        assert [b.text for b in doc.blocks] == ["A.", "B."]
        assert doc.location_kind is None
        assert all(b.location is None for b in doc.blocks)

    def test_line_numbers_and_crlf(self) -> None:
        doc = parse_markdown("# T\r\n\r\nOne.\r\n\r\nTwo\r\nlines.\r\n")

        assert [(b.text, b.line) for b in doc.blocks] == [("One.", 2), ("Two\nlines.", 4)]

    # ---- real extractor output ---- #

    async def test_pdf_pages_become_locations(self) -> None:
        doc = parse_markdown(await _extracted("widget-spec.pdf", "application/pdf"))

        assert doc.location_kind == "page"
        assert {b.location for b in doc.blocks} == {1, 2}
        spec = next(b for b in doc.blocks if "1200 items per minute" in b.text)
        assert spec.location == 1
        assert not any("<!--" in b.text for b in doc.blocks)

    async def test_pptx_slides_are_sections_with_locations(self) -> None:
        mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        doc = parse_markdown(await _extracted("roadmap.pptx", mime))

        assert doc.location_kind == "slide"
        titled = [s for s in doc.sections if s.level == 2]
        assert len(titled) >= 3
        assert [s.location for s in titled] == sorted(s.location or 0 for s in titled)
        assert all(s.location is not None for s in titled)

    async def test_xlsx_sheets_give_tables_with_locations(self) -> None:
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        doc = parse_markdown(await _extracted("pricing.xlsx", mime))

        tables = [b for b in doc.blocks if b.kind is BlockKind.TABLE]
        assert doc.location_kind == "sheet"
        assert {t.location for t in tables} == {1, 2}
        assert all(t.header and t.header.count("\n") == 1 and t.parts for t in tables)
