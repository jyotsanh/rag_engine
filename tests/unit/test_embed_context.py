"""Contextual embed text: page title + heading path in front of each chunk, bounded."""

import random

import pytest
from pydantic import ValidationError

from rag_engine.chunking import (
    ChunkingConfig,
    ChunkInput,
    StructureChunker,
    build_context,
    compose_embed_text,
    tokenizer_for,
)
from rag_engine.config import Settings
from rag_engine.domain import ContentKind

_TOKENIZER = tokenizer_for("text-embedding-3-small")


def _context(title: str | None, *path: str, budget: int = 100) -> str:
    return build_context(title, path, tokenizer=_TOKENIZER, max_tokens=budget)


def _source(markdown: str, title: str | None = "Install guide | Acme Widgets") -> ChunkInput:
    return ChunkInput(
        tenant_id="ten_" + "1" * 32,
        kb_id="kb_" + "1" * 32,
        source_id="src_" + "1" * 32,
        version="job_" + "1" * 32,
        page_id="page_" + "1" * 32,
        url="https://acme.example/install",
        markdown=markdown,
        content_kind=ContentKind.HTML,
        title=title,
    )


# ---- #
# Tests
# ---- #


class TestContextualEmbedText_S58:
    """S58 — Contextual embed text: title + heading path + chunk; `text` stays clean."""

    # ---- the context ---- #

    def test_title_and_path_on_their_own_lines(self) -> None:
        context = _context("Acme Widgets", "Getting started", "Linux")

        assert context == "Acme Widgets\nGetting started > Linux"
        assert compose_embed_text(context, "Run it.") == (
            "Acme Widgets\nGetting started > Linux\n\nRun it."
        )

    @pytest.mark.parametrize(
        ("title", "path", "expected"),
        [
            (None, (), ""),
            ("  Acme   Widgets ", (), "Acme Widgets"),  # whitespace normalized
            (None, ("Setup", "Linux"), "Setup > Linux"),
            ("Acme", ("", "  ", "Setup"), "Acme\nSetup"),  # blank headings ignored
        ],
    )
    def test_missing_parts(self, title: str | None, path: tuple[str, ...], expected: str) -> None:
        assert _context(title, *path) == expected
        assert compose_embed_text("", "Run it.") == "Run it."

    @pytest.mark.parametrize(
        ("title", "heading", "dropped"),
        [
            ("Install guide", "Install guide", True),
            ("install  GUIDE", "Install guide", True),  # case/space-insensitive
            ("Install guide | Acme Widgets", "Install guide", True),
            ("Acme Widgets - Install guide", "Install guide", True),
            ("Acme — Install guide", "Install guide", True),
            ("Docs: Install guide", "Install guide", True),
            ("WX-2048 manual", "WX", False),  # a hyphen inside a word is not a separator
            ("Install guide for Linux", "Install guide", False),  # only whole parts
        ],
    )
    def test_a_heading_repeating_the_title_is_dropped(
        self, title: str, heading: str, dropped: bool
    ) -> None:
        context = _context(title, heading, "Linux")

        expected_path = "Linux" if dropped else f"{heading} > Linux"
        assert context == f"{' '.join(title.split())}\n{expected_path}"

    # ---- the budget ---- #

    def test_a_deep_path_collapses_its_middle_first(self) -> None:
        path = ["Reference", *(f"Level {i} with a long descriptive name" for i in range(8)), "Leaf"]

        context = _context("Acme", *path, budget=20)

        assert context == "Acme\nReference > … > Leaf"

    def test_a_long_title_is_shortened_with_an_ellipsis(self) -> None:
        title = "The complete and unabridged guide " * 10

        context = _context(title, "Setup", budget=20)

        first, second = context.split("\n")
        assert first.startswith("The complete and unabridged guide") and first.endswith("…")
        assert second == "Setup"
        assert _TOKENIZER.count(context) <= 20

    def test_long_headings_are_shortened_too(self) -> None:
        context = _context("Acme", "word " * 80, "other " * 80, budget=25)

        assert _TOKENIZER.count(context) <= 25
        assert context.startswith("Acme\n") and "…" in context

    def test_a_zero_budget_disables_the_context(self) -> None:
        assert _context("Acme", "Setup", budget=0) == ""

    def test_any_title_and_path_fit_the_budget(self) -> None:
        rng = random.Random(58)
        vocabulary = ["widget", "WX-2048", "Übersicht", "設定", "a", "x" * 60, "sorter", "|"]
        for _ in range(200):
            title = " ".join(rng.choice(vocabulary) for _ in range(rng.randint(0, 40)))
            path = [
                " ".join(rng.choice(vocabulary) for _ in range(rng.randint(1, 15)))
                for _ in range(rng.randint(0, 12))
            ]
            budget = rng.randint(1, 60)

            context = _context(title or None, *path, budget=budget)

            assert _TOKENIZER.count(context) <= budget

    # ---- in the chunker ---- #

    def test_chunks_embed_their_context_and_display_clean_text(self) -> None:
        chunker = StructureChunker(_TOKENIZER, ChunkingConfig.from_settings())
        source = _source("# Install guide\n\nIntro.\n\n## Linux\n\nRun the installer.\n")

        drafts = chunker.chunk(source)

        assert [(d.text, d.embed_text) for d in drafts] == [
            ("Intro.", "Install guide | Acme Widgets\n\nIntro."),  # h1 repeats the title
            ("Run the installer.", "Install guide | Acme Widgets\nLinux\n\nRun the installer."),
        ]
        assert all(d.token_count == _TOKENIZER.count(d.embed_text) for d in drafts)

    def test_the_same_text_in_another_section_embeds_differently(self) -> None:
        chunker = StructureChunker(_TOKENIZER, ChunkingConfig.from_settings())

        a, b = chunker.chunk(_source("# Linux\n\nRestart.\n\n# Windows\n\nRestart.\n"))

        assert a.text == b.text == "Restart."
        assert a.content_hash != b.content_hash  # the hash covers what is embedded

    def test_a_page_without_title_or_headings_embeds_its_text(self) -> None:
        chunker = StructureChunker(_TOKENIZER, ChunkingConfig.from_settings())

        (draft,) = chunker.chunk(_source("Just a paragraph.\n", title=None))

        assert draft.embed_text == draft.text == "Just a paragraph."

    def test_long_context_never_breaks_the_cap(self) -> None:
        config = ChunkingConfig(target_tokens=60, max_tokens=120, overlap_tokens=10)
        chunker = StructureChunker(_TOKENIZER, config)
        heading = " ".join(["Configuring"] * 60)
        prose = " ".join(f"Sentence {i} explains widget setting number {i}." for i in range(60))

        drafts = chunker.chunk(_source(f"# {heading}\n\n## {heading}\n\n{prose}\n", "T " * 300))

        assert config.context_tokens == 60  # min(100, 120 - 60)
        assert len(drafts) > 3
        assert all(d.token_count <= config.max_tokens for d in drafts)
        assert all(d.embed_text.endswith(d.text) for d in drafts)

    def test_stored_chunks_keep_both_texts(self) -> None:
        source = _source("# Linux\n\nRun the installer.\n")
        (draft,) = StructureChunker(_TOKENIZER, ChunkingConfig.from_settings()).chunk(source)

        chunk = draft.to_document(source)

        assert chunk.text == "Run the installer."
        assert chunk.embed_text == "Install guide | Acme Widgets\nLinux\n\nRun the installer."
        assert (chunk.title, chunk.heading_path) == ("Install guide | Acme Widgets", ["Linux"])

    # ---- configuration ---- #

    def test_context_budget_config(self) -> None:
        assert ChunkingConfig(400, 800, 60).context_tokens == 100
        assert ChunkingConfig(400, 450, 60).context_tokens == 50  # what the cap leaves
        assert ChunkingConfig(400, 800, 60, context_tokens=0).context_budget == 0
        assert (
            ChunkingConfig.from_settings(
                Settings(_env_file=None, CHUNK_CONTEXT_MAX_TOKENS=40)
            ).context_tokens
            == 40
        )
        with pytest.raises(ValueError, match="context_tokens"):
            ChunkingConfig(400, 450, 60, context_tokens=51)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"CHUNK_CONTEXT_MAX_TOKENS": 401},  # 800 - 400 leaves 400
            {"CHUNK_CONTEXT_MAX_TOKENS": -1},
        ],
    )
    def test_context_settings_are_validated(self, overrides: dict[str, int]) -> None:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **overrides)  # ty: ignore[invalid-argument-type]
