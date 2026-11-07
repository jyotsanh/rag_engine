"""Page outcomes: every field written, Markdown capped at a block boundary."""

import pytest

from rag_engine.domain import ContentKind, PageReason, PageStatus
from rag_engine.repositories import PageOutcome, truncate_markdown

# ---- #
# Tests
# ---- #


class TestPageOutcome_S47:
    """S47 — Page repository: outcome values and Markdown truncation (no database)."""

    def test_changes_set_every_outcome_field(self) -> None:
        outcome = PageOutcome(status=PageStatus.SKIPPED, reason=PageReason.OUT_OF_SCOPE)

        changes = outcome.changes(max_markdown_bytes=1000)

        # a skip after an earlier success must clear the old content fields
        assert changes["markdown"] is None and changes["content_kind"] is None
        assert changes["title"] is None and changes["duplicate_of"] is None
        assert changes["js_signals"] == [] and changes["doc_metadata"] == {}
        assert changes["markdown_truncated"] is False
        assert {"status", "reason", "reason_detail", "fetch_ms", "extract_ms"} <= set(changes)

    def test_reason_detail_is_capped(self) -> None:
        outcome = PageOutcome(
            status=PageStatus.FAILED, reason=PageReason.EXTRACTION_FAILED, reason_detail="x" * 900
        )

        assert len(outcome.changes(1000)["reason_detail"]) == 500

    def test_markdown_over_the_cap_is_truncated_and_flagged(self) -> None:
        markdown = "# Title\n\n" + "\n\n".join(f"Paragraph {n} " + "w" * 40 for n in range(50))
        outcome = PageOutcome(
            status=PageStatus.SUCCEEDED, content_kind=ContentKind.HTML, markdown=markdown
        )

        changes = outcome.changes(max_markdown_bytes=500)

        assert changes["markdown_truncated"] is True
        assert len(changes["markdown"].encode()) <= 500
        assert changes["markdown"].endswith("w\n")  # ends on a whole paragraph
        assert markdown.startswith(changes["markdown"].rstrip("\n"))

    @pytest.mark.parametrize(
        ("markdown", "max_bytes", "expected"),
        [
            (None, 10, (None, False)),
            ("short", 10, ("short", False)),
            ("a" * 10, 10, ("a" * 10, False)),  # exactly at the cap
            ("one\n\ntwo\n\nthree", 10, ("one\n\ntwo\n", True)),  # "two" + blank line fit
            ("one\n\ntwo\n\nthree", 9, ("one\n", True)),  # no blank line after "two" yet
            ("no-boundary-at-all", 8, ("no-bound\n", True)),
            ("é" * 6, 5, ("éé\n", True)),  # never splits a multi-byte character
        ],
    )
    def test_truncate_markdown(
        self, markdown: str | None, max_bytes: int, expected: tuple[str | None, bool]
    ) -> None:
        assert truncate_markdown(markdown, max_bytes) == expected
