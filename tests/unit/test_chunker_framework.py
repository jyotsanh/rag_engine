"""Chunker framework: the base contract every strategy inherits, the factory, settings."""

import hashlib
from collections.abc import Iterable, Iterator

import pytest
from pydantic import ValidationError

from rag_engine.chunking import (
    MAX_HEADING_DEPTH,
    BaseChunker,
    ChunkerRegistrationError,
    ChunkingConfig,
    ChunkInput,
    ChunkPiece,
    ChunkTooLargeError,
    TokenizerError,
    UnknownChunkerError,
    get_chunker,
    register_chunker,
    registered_chunkers,
)
from rag_engine.chunking import factory as chunker_factory
from rag_engine.config import ChunkerType, Settings
from rag_engine.domain import Chunk, ContentKind, Page, PageReason, PageStatus

_IDS = {
    "tenant_id": "ten_" + "1" * 32,
    "kb_id": "kb_" + "1" * 32,
    "source_id": "src_" + "1" * 32,
    "job_id": "job_" + "1" * 32,
}


class _WordTokenizer:
    """One token per whitespace-separated word (tiktoken arrives in Step 57)."""

    name = "words"

    def count(self, text: str) -> int:
        return len(text.split())


class _ZeroTokenizer:
    name = "broken"

    def count(self, text: str) -> int:
        return 0


class _ScriptedChunker(BaseChunker):
    """Returns whatever pieces it was given; the base class does the rest."""

    name = "scripted"
    pieces: tuple[ChunkPiece, ...] = ()

    def split(self, source: ChunkInput) -> Iterable[ChunkPiece]:
        return list(self.pieces)


def _page(**changes: object) -> Page:
    fields: dict[str, object] = {
        **_IDS,
        "url": "https://acme.example/docs/",
        "url_hash": "0" * 64,
        "status": PageStatus.SUCCEEDED,
        "content_kind": ContentKind.HTML,
        "markdown": "# Docs\n\nWidgets sort things.\n",
        "title": "Docs | Acme",
        "lang": "en",
        **changes,
    }
    return Page.model_validate(fields)


def _source(markdown: str = "# Docs\n\nText.\n") -> ChunkInput:
    return ChunkInput.from_page(_page(markdown=markdown))


def _chunker(
    *pieces: ChunkPiece, tokenizer: object = None, max_tokens: int = 50
) -> _ScriptedChunker:
    chunker = _ScriptedChunker(
        tokenizer or _WordTokenizer(),  # ty: ignore[invalid-argument-type]
        ChunkingConfig(target_tokens=min(20, max_tokens), max_tokens=max_tokens, overlap_tokens=2),
    )
    chunker.pieces = pieces
    return chunker


@pytest.fixture
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(chunker_factory, "_CHUNKERS", {})
    yield


# ---- #
# Tests
# ---- #


class TestChunkerFramework_S55:
    """S55 — Chunker framework: `BaseChunker` contract, `ChunkInput`/`ChunkDraft`, factory."""

    # ---- input ---- #

    def test_input_is_built_from_a_succeeded_page(self) -> None:
        page = _page()

        source = ChunkInput.from_page(page)
        stripped = ChunkInput.from_page(page, markdown="# Docs\n\nOnly this.\n")

        assert (source.version, source.page_id, source.url) == (_IDS["job_id"], page.id, page.url)
        assert (source.title, source.lang, source.content_kind) == ("Docs | Acme", "en", "html")
        assert source.markdown == page.markdown
        assert stripped.markdown == "# Docs\n\nOnly this.\n"

    def test_only_succeeded_pages_are_chunked(self) -> None:
        skipped = _page(
            status=PageStatus.SKIPPED,
            reason=PageReason.NOINDEX,
            markdown=None,
            content_kind=None,
        )

        with pytest.raises(ValueError, match="only succeeded pages"):
            ChunkInput.from_page(skipped)

    # ---- the base contract ---- #

    def test_pieces_become_dense_ordered_drafts(self) -> None:
        chunker = _chunker(
            ChunkPiece("\n\nFirst piece here.\n\n", ("Docs",)),
            ChunkPiece("   \n\n"),  # empty: dropped, no gap in ordinals
            ChunkPiece("    indented code\n    stays   \n", ("Docs", "Code"), page_number=3),
        )

        drafts = chunker.chunk(_source())

        assert [d.ordinal for d in drafts] == [0, 1]
        assert drafts[0].text == "First piece here."
        assert drafts[1].text == "    indented code\n    stays"  # indentation kept
        assert drafts[1].heading_path == ("Docs", "Code")
        assert (drafts[0].page_number, drafts[1].page_number) == (None, 3)

    def test_tokens_and_hash_describe_the_embedded_text(self) -> None:
        class _Contextual(_ScriptedChunker):
            def embed_text(self, source: ChunkInput, piece: ChunkPiece, text: str) -> str:
                return f"{source.title}\n{text}"

        chunker = _Contextual(
            _WordTokenizer(), ChunkingConfig(target_tokens=10, max_tokens=20, overlap_tokens=0)
        )
        chunker.pieces = (ChunkPiece("three words here"),)

        (draft,) = chunker.chunk(_source())

        assert draft.text == "three words here"
        assert draft.embed_text == "Docs | Acme\nthree words here"
        assert draft.token_count == 6  # counted on embed_text
        assert draft.content_hash == hashlib.sha256(draft.embed_text.encode()).hexdigest()

    def test_same_embedded_text_same_hash(self) -> None:
        drafts = _chunker(ChunkPiece("same words"), ChunkPiece("same words\n")).chunk(_source())

        assert drafts[0].content_hash == drafts[1].content_hash
        assert drafts[0].ordinal != drafts[1].ordinal

    def test_heading_paths_are_capped(self) -> None:
        path = tuple(f"H{n}" for n in range(20))

        (draft,) = _chunker(ChunkPiece("text", path)).chunk(_source())

        assert draft.heading_path == path[:MAX_HEADING_DEPTH]

    def test_an_oversized_chunk_is_a_splitter_bug(self) -> None:
        chunker = _chunker(ChunkPiece("ok"), ChunkPiece("word " * 11), max_tokens=10)

        with pytest.raises(ChunkTooLargeError) as exc:
            chunker.chunk(_source())

        assert (exc.value.ordinal, exc.value.token_count, exc.value.max_tokens) == (1, 11, 10)
        assert exc.value.url == "https://acme.example/docs/"

    def test_a_zero_token_count_is_refused(self) -> None:
        chunker = _chunker(ChunkPiece("text"), tokenizer=_ZeroTokenizer())

        with pytest.raises(TokenizerError, match="broken counted 0"):
            chunker.chunk(_source())

    def test_no_pieces_no_chunks(self) -> None:
        assert _chunker().chunk(_source()) == []

    # ---- persistence shape ---- #

    def test_draft_becomes_a_valid_chunk_document(self) -> None:
        source = ChunkInput.from_page(_page(title="T" * 1000))
        (draft,) = _chunker(ChunkPiece("Widgets sort things.", ("Docs",), 2)).chunk(source)

        chunk = draft.to_document(source)
        restored = Chunk.from_mongo(chunk.to_mongo())

        assert restored == chunk
        assert (chunk.version, chunk.page_id, chunk.url) == (
            source.version,
            source.page_id,
            source.url,
        )
        assert (chunk.ordinal, chunk.heading_path, chunk.page_number) == (0, ["Docs"], 2)
        assert chunk.title is not None and len(chunk.title) == 1000
        assert chunk.id.startswith("chk_")

    def test_chunk_page_number_is_positive(self) -> None:
        source = _source()
        (draft,) = _chunker(ChunkPiece("text")).chunk(source)
        chunk = draft.to_document(source)

        with pytest.raises(ValidationError):
            Chunk.model_validate({**chunk.model_dump(), "page_number": 0})

    # ---- config ---- #

    def test_config_comes_from_settings_and_is_validated(self) -> None:
        config = ChunkingConfig.from_settings(
            Settings(
                _env_file=None,
                CHUNK_TARGET_TOKENS=300,
                CHUNK_MAX_TOKENS=600,
                CHUNK_OVERLAP_TOKENS=30,
            )
        )

        assert (config.target_tokens, config.max_tokens, config.overlap_tokens) == (300, 600, 30)
        with pytest.raises(ValueError, match="target_tokens <= max_tokens"):
            ChunkingConfig(target_tokens=10, max_tokens=5, overlap_tokens=0)
        with pytest.raises(ValueError, match="overlap_tokens < target_tokens"):
            ChunkingConfig(target_tokens=10, max_tokens=20, overlap_tokens=10)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"CHUNK_TARGET_TOKENS": 900, "CHUNK_MAX_TOKENS": 800},
            {"CHUNK_OVERLAP_TOKENS": 400, "CHUNK_TARGET_TOKENS": 400},
            {"CHUNK_MAX_TOKENS": 9000},  # over the embedding models' input limit
            {"CHUNK_TARGET_TOKENS": 0},
            {"CHUNKER": "sentences"},
        ],
    )
    def test_inconsistent_chunk_settings_are_rejected(self, overrides: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **overrides)  # ty: ignore[invalid-argument-type]

    # ---- factory ---- #

    def test_factory_builds_the_registered_chunker(self, isolated_registry: None) -> None:
        register_chunker(ChunkerType.STRUCTURE)(_ScriptedChunker)
        tokenizer = _WordTokenizer()
        budget = ChunkingConfig(target_tokens=5, max_tokens=9, overlap_tokens=1)

        default = get_chunker(tokenizer)
        explicit = get_chunker(tokenizer, ChunkerType.STRUCTURE, chunking=budget)

        assert isinstance(default, _ScriptedChunker) and default.tokenizer is tokenizer
        assert default.config == ChunkingConfig.from_settings()
        assert explicit.config == budget
        assert registered_chunkers() == {ChunkerType.STRUCTURE: _ScriptedChunker}

    def test_an_unregistered_strategy_is_a_typed_error(self, isolated_registry: None) -> None:
        with pytest.raises(UnknownChunkerError, match="no chunker registered for 'structure'"):
            get_chunker(_WordTokenizer(), ChunkerType.STRUCTURE)

    def test_registration_is_checked(self, isolated_registry: None) -> None:
        class _Other(_ScriptedChunker):
            pass

        register_chunker(ChunkerType.STRUCTURE)(_ScriptedChunker)
        register_chunker(ChunkerType.STRUCTURE)(_ScriptedChunker)  # same class again: fine

        with pytest.raises(ChunkerRegistrationError, match="already registered"):
            register_chunker(ChunkerType.STRUCTURE)(_Other)
        with pytest.raises(ChunkerRegistrationError, match="not a BaseChunker"):
            register_chunker(ChunkerType.STRUCTURE)(dict)  # ty: ignore[invalid-argument-type]
