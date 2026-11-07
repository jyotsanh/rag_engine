"""Token-aware structure chunker (tiktoken) and the model tokenizer."""

import itertools
import random
import re

import pytest
import tiktoken

from rag_engine.chunking import (
    ChunkDraft,
    ChunkingConfig,
    ChunkInput,
    ChunkPiece,
    StructureChunker,
    TiktokenTokenizer,
    get_chunker,
    sentences,
    tokenizer_for,
)
from rag_engine.config import EmbeddingModelName
from rag_engine.domain import ContentKind
from rag_engine.embeddings import UnsupportedEmbeddingModelError
from rag_engine.extraction import ExtractionInput, default_registry
from tests.support.fixture_site import SITE_ROOT

_TOKENIZER = tokenizer_for(EmbeddingModelName.TEXT_EMBEDDING_3_SMALL)
_SMALL = ChunkingConfig(target_tokens=60, max_tokens=120, overlap_tokens=20)


def _source(markdown: str, *, title: str | None = "Widgets") -> ChunkInput:
    return ChunkInput(
        tenant_id="ten_" + "1" * 32,
        kb_id="kb_" + "1" * 32,
        source_id="src_" + "1" * 32,
        version="job_" + "1" * 32,
        page_id="page_" + "1" * 32,
        url="https://acme.example/guide",
        markdown=markdown,
        content_kind=ContentKind.HTML,
        title=title,
    )


def _chunk(markdown: str, config: ChunkingConfig = _SMALL) -> list[ChunkDraft]:
    return StructureChunker(_TOKENIZER, config).chunk(_source(markdown))


def _count(text: str) -> int:
    return _TOKENIZER.count(text)


def _prose(n: int, start: int = 0) -> str:
    return " ".join(
        f"Sentence {i} says the WX-{2000 + i} sorter handles {i * 10} widgets per minute."
        for i in range(start, start + n)
    )


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9-]+", text)


def _fixture(name: str) -> bytes:
    return (SITE_ROOT / "files" / name).read_bytes()


# ---- #
# Tests
# ---- #


class TestTokenAwareSplitter_S57:
    """S57 — Token-aware splitter: tiktoken budgets, structure-first recursive splitting."""

    # ---- tokenizer ---- #

    def test_tokenizer_counts_like_the_model_encoding(self) -> None:
        text = "The WX-2048 sorts 1200 items per minute — très bien. 日本語も。"

        assert _TOKENIZER.name == "cl100k_base"
        assert _TOKENIZER.count(text) == len(tiktoken.get_encoding("cl100k_base").encode(text))
        assert _TOKENIZER.decode(_TOKENIZER.encode(text)) == text

    def test_tokenizer_is_shared_per_encoding(self) -> None:
        small = tokenizer_for(EmbeddingModelName.TEXT_EMBEDDING_3_SMALL)
        large = tokenizer_for("text-embedding-3-large")

        assert small is large  # both models use cl100k_base
        assert isinstance(small, TiktokenTokenizer)
        with pytest.raises(UnsupportedEmbeddingModelError):
            tokenizer_for("text-embedding-ada-002")

    def test_special_token_text_on_a_page_is_plain_text(self) -> None:
        assert _TOKENIZER.count("before <|endoftext|> after") > 3

    def test_structure_is_the_default_chunker(self) -> None:
        chunker = get_chunker(_TOKENIZER)

        assert isinstance(chunker, StructureChunker)
        assert chunker.config == ChunkingConfig.from_settings()

    # ---- sentences ---- #

    def test_sentences_respect_abbreviations_quotes_and_numbers(self) -> None:
        text = (
            'He said "stop." Then e.g. this works. J. Smith left! Is it 1.5 m? Yes… ok. '
            "Fig. 3 shows it. Done。次の文。"
        )

        assert sentences(text) == [
            'He said "stop."',
            "Then e.g. this works.",
            "J. Smith left!",
            "Is it 1.5 m?",
            "Yes…",
            "ok.",
            "Fig. 3 shows it.",
            "Done。",
            "次の文。",
        ]

    # ---- sections and packing ---- #

    def test_small_sections_become_one_chunk_each(self) -> None:
        drafts = _chunk(
            "Intro.\n\n# Guide\n\nAbout widgets.\n\n## Install\n\nRun the installer.\n\n"
            "## Empty\n\n## Configure\n\nSet the speed.\n"
        )

        assert [(d.heading_path, d.text) for d in drafts] == [
            ((), "Intro."),
            (("Guide",), "About widgets."),
            (("Guide", "Install"), "Run the installer."),
            (("Guide", "Configure"), "Set the speed."),  # the empty section has no chunk
        ]
        assert not any("#" in d.text for d in drafts)  # headings live in heading_path

    def test_paragraphs_pack_up_to_the_target(self) -> None:
        paragraphs = [f"Paragraph {i} about widget model {i}." for i in range(40)]

        drafts = _chunk("# Many\n\n" + "\n\n".join(paragraphs) + "\n")

        assert len(drafts) > 1
        assert all(d.token_count <= _SMALL.target_tokens for d in drafts)
        # each paragraph appears whole, once, in order (no overlap between whole paragraphs)
        joined = [p for d in drafts for p in d.text.split("\n\n")]
        assert joined == paragraphs

    def test_long_prose_splits_at_sentences_with_overlap(self) -> None:
        text = _prose(20)

        drafts = _chunk(f"# Speeds\n\n{text}\n")

        assert len(drafts) > 2
        for previous, current in itertools.pairwise(drafts):
            last = sentences(previous.text)[-1]
            assert current.text.startswith(last)  # the overlap: whole trailing sentence(s)
            assert _count(last) <= _SMALL.overlap_tokens
        all_sentences = [s for d in drafts for s in sentences(d.text)]
        assert list(dict.fromkeys(all_sentences)) == sentences(text)  # nothing lost, in order
        assert all(d.token_count <= _SMALL.max_tokens for d in drafts)

    def test_no_overlap_across_blocks_or_sections(self) -> None:
        after = _prose(2, 50)
        markdown = (
            f"# A\n\n{_prose(12)}\n\n| h |\n|---|\n| v |\n\n{after}\n\n# B\n\n{_prose(2, 90)}\n"
        )

        drafts = _chunk(markdown)

        everything = "\n".join(d.text for d in drafts)
        assert everything.count("| h |") == 1  # the table is never repeated
        for sentence in sentences(after):  # the block after the table: not overlapped
            assert everything.count(sentence) == 1
        b = next(d for d in drafts if d.heading_path == ("B",))
        assert b.text.startswith("Sentence 90")  # nothing carried from section A

    def test_overlap_can_be_disabled(self) -> None:
        config = ChunkingConfig(target_tokens=60, max_tokens=120, overlap_tokens=0)

        drafts = _chunk(f"# S\n\n{_prose(20)}\n", config)

        total = sum(len(sentences(d.text)) for d in drafts)
        assert total == 20

    # ---- atomic blocks ---- #

    def test_an_atomic_block_over_target_but_under_the_cap_stays_whole(self) -> None:
        rows = "\n".join(f"| WX-{i} | {i * 100} items |" for i in range(9))
        table = f"| Model | Speed |\n|---|---|\n{rows}"
        assert _SMALL.target_tokens < _count(table) <= _SMALL.max_tokens

        drafts = _chunk(f"# T\n\nBefore.\n\n{table}\n\nAfter.\n")

        assert [d.text for d in drafts] == ["Before.", table, "After."]

    def test_an_oversized_table_repeats_its_header(self) -> None:
        rows = [f"| WX-{i} | {i * 100} items per minute | rev {i} |" for i in range(60)]
        header = "| Model | Speed | Revision |\n| --- | --- | --- |"

        drafts = _chunk(f"# T\n\n{header}\n" + "\n".join(rows) + "\n")

        assert len(drafts) > 2
        seen: list[str] = []
        for draft in drafts:
            lines = draft.text.split("\n")
            assert "\n".join(lines[:2]) == header
            seen.extend(lines[2:])
            assert draft.token_count <= _SMALL.max_tokens
        assert seen == rows  # every row once, in order

    def test_oversized_code_stays_fenced_line_by_line(self) -> None:
        lines = [f"    value_{i} = compute(widget_{i}, speed={i * 10})" for i in range(50)]

        drafts = _chunk("# Code\n\n```python\n" + "\n".join(lines) + "\n```\n")

        assert len(drafts) > 1
        body: list[str] = []
        for draft in drafts:
            text = draft.text.split("\n")
            assert (text[0], text[-1]) == ("```python", "```")
            body.extend(text[1:-1])
        assert body == lines  # indentation intact

    def test_an_unclosed_fence_is_closed_with_its_own_fence(self) -> None:
        lines = [f"step {i}: move widget {i} to bin {i % 7}" for i in range(40)]

        drafts = _chunk("# Log\n\n~~~~\n" + "\n".join(lines) + "\n")

        assert len(drafts) > 1
        assert all(d.text.startswith("~~~~\n") and d.text.endswith("\n~~~~") for d in drafts)

    def test_an_oversized_list_splits_between_items(self) -> None:
        items = [f"- Item {i}: the WX-{i} needs part {i * 3}\n  - detail {i}" for i in range(30)]

        drafts = _chunk("# Parts\n\n" + "\n".join(items) + "\n")

        assert len(drafts) > 1
        assert [d.text for d in drafts] == [d.text for d in drafts if d.text.startswith("- Item")]
        assert "\n".join(d.text for d in drafts) == "\n".join(items)

    # ---- the last resorts ---- #

    def test_lines_without_punctuation_split_by_line(self) -> None:
        lines = [f"Acme Widgets warehouse {i} Industrial Road {i} Springfield" for i in range(30)]

        no_overlap = ChunkingConfig(target_tokens=60, max_tokens=120, overlap_tokens=0)

        drafts = _chunk("# Addresses\n\n" + "  \n".join(lines) + "\n", no_overlap)  # hard breaks

        assert len(drafts) > 1
        assert all(d.token_count <= _SMALL.max_tokens for d in drafts)
        assert _words(" ".join(d.text for d in drafts)) == _words(" ".join(lines))

    def test_a_single_huge_token_run_is_cut_by_characters(self) -> None:
        blob = "".join(random.Random(7).choice("abcdefghijklmnop0123456789+/") for _ in range(3000))

        drafts = _chunk(f"# Blob\n\n{blob}\n")

        assert len(drafts) > 1
        assert all(d.token_count <= _SMALL.max_tokens for d in drafts)
        assert "".join(d.text for d in drafts) == blob  # nothing lost or added

    # ---- budgets and locations ---- #

    def test_the_embed_text_context_is_reserved_within_the_cap(self) -> None:
        class _Contextual(StructureChunker):
            def embed_text(self, source: ChunkInput, piece: ChunkPiece, text: str) -> str:
                prefix = " > ".join((source.title or "", *piece.heading_path))
                return f"{prefix}\n\n{text}"

        long_path = "# " + " ".join(["Widget"] * 25) + "\n\n## " + " ".join(["Speed"] * 25)
        chunker = _Contextual(_TOKENIZER, _SMALL)

        drafts = chunker.chunk(_source(f"{long_path}\n\n{_prose(30)}\n"))

        assert all(d.token_count <= _SMALL.max_tokens for d in drafts)
        assert all(d.embed_text.startswith("Widgets > Widget") for d in drafts)

    def test_chunks_carry_the_location_of_their_first_block(self) -> None:
        drafts = _chunk(
            "<!-- page 1 -->\n# Spec\n\nOn page one.\n\n<!-- page 2 -->\n## Next\n\n"
            "<!-- page 3 -->\nOn page three.\n"
        )

        assert [(d.heading_path, d.page_number) for d in drafts] == [
            (("Spec",), 1),
            (("Spec", "Next"), 3),
        ]

    def test_random_documents_never_break_the_contract(self) -> None:
        rng = random.Random(56)
        blocks = [
            lambda: _prose(rng.randint(1, 15), rng.randint(0, 500)),
            lambda: (
                "| a | b |\n|---|---|\n"
                + "\n".join(f"| r{i} | {_prose(1, i)} |" for i in range(rng.randint(1, 25)))
            ),
            lambda: "```\n" + "\n".join(f"x{i} = {i}" for i in range(rng.randint(1, 80))) + "\n```",
            lambda: "\n".join(f"- item {i} {_prose(1, i)}" for i in range(rng.randint(1, 20))),
            lambda: f"{'#' * rng.randint(1, 4)} Heading {rng.randint(0, 99)}",
        ]
        for _ in range(25):
            markdown = "\n\n".join(rng.choice(blocks)() for _ in range(rng.randint(1, 12)))

            drafts = _chunk(markdown)

            assert [d.ordinal for d in drafts] == list(range(len(drafts)))
            assert all(0 < d.token_count <= _SMALL.max_tokens for d in drafts)
            source_words = set(_words(re.sub(r"(?m)^#+ .*$", "", markdown)))
            assert source_words <= set(_words(" ".join(d.text for d in drafts)))

    # ---- real documents ---- #

    @pytest.mark.parametrize(
        ("name", "mime", "located"),
        [
            ("widget-spec.pdf", "application/pdf", True),
            ("install-guide.docx",
             "application/vnd.openxmlformats-officedocument.wordprocessingml.document", False),
            ("roadmap.pptx",
             "application/vnd.openxmlformats-officedocument.presentationml.presentation", True),
            ("pricing.xlsx",
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", True),
        ],
    )  # fmt: skip
    async def test_fixture_documents_chunk_within_budget(
        self, name: str, mime: str, located: bool
    ) -> None:
        body = _fixture(name)
        document = await default_registry().extract(
            ExtractionInput(url=f"https://acme.example/files/{name}", mime_type=mime, body=body)
        )

        drafts = StructureChunker(_TOKENIZER, ChunkingConfig.from_settings()).chunk(
            _source(document.markdown, title=document.title)
        )

        assert drafts
        assert all(0 < d.token_count <= ChunkingConfig.from_settings().max_tokens for d in drafts)
        assert not any("<!--" in d.text for d in drafts)
        assert all((d.page_number is not None) is located for d in drafts)
