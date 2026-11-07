"""Chunk persistence in real MongoDB: per-page idempotent writes, version reads/cleanup."""

from collections.abc import AsyncIterator

import pytest

from rag_engine.chunking import ChunkingConfig, ChunkInput, StructureChunker, tokenizer_for
from rag_engine.config import Settings
from rag_engine.domain import Chunk, ContentKind, Page, PageStatus, chunk_id
from rag_engine.infrastructure import MongoConnection
from rag_engine.repositories import ChunkRepository

_TENANT = "ten_" + "1" * 32
_KB = "kb_" + "1" * 32
_SOURCE = "src_" + "1" * 32
_OTHER_SOURCE = "src_" + "2" * 32
_V1 = "job_" + "1" * 32
_V2 = "job_" + "2" * 32
_CHUNKER = StructureChunker(
    tokenizer_for("text-embedding-3-small"),
    ChunkingConfig(target_tokens=40, max_tokens=120, overlap_tokens=0),
)


def _page(n: int, markdown: str, *, version: str = _V1, source: str = _SOURCE) -> Page:
    return Page(
        tenant_id=_TENANT,
        kb_id=_KB,
        source_id=source,
        job_id=version,
        url=f"https://acme.example/p{n}",
        url_hash=f"{n:064x}",
        status=PageStatus.SUCCEEDED,
        content_kind=ContentKind.HTML,
        markdown=markdown,
        title=f"Page {n}",
    )


def _markdown(sections: int) -> str:
    return "\n\n".join(
        f"## Section {i}\n\nThe WX-{i} sorter moves {i * 100} widgets per minute."
        for i in range(sections)
    )


def _chunks(page: Page, markdown: str | None = None) -> list[Chunk]:
    return _CHUNKER.chunk_documents(ChunkInput.from_page(page, markdown=markdown))


@pytest.fixture
async def repo(integration_settings: Settings) -> AsyncIterator[ChunkRepository]:
    mongo = MongoConnection(integration_settings)
    await mongo.connect()
    await mongo.ensure_indexes([Chunk])
    yield ChunkRepository(mongo.db)
    await mongo.close()


# ---- #
# Tests
# ---- #


class TestChunkPersistence_S59:
    """S59 — Chunk persistence: per (source, version), idempotent per page, for re-embedding."""

    async def test_a_chunked_page_is_stored_and_read_back_in_order(
        self, repo: ChunkRepository
    ) -> None:
        page = _page(1, _markdown(4))
        chunks = _chunks(page)

        saved = await repo.save_page_chunks(page.id, chunks)
        stored = await repo.for_page(page.id)

        assert saved == len(chunks) == 4
        assert stored == chunks  # every field round-trips, in ordinal order
        assert [c.id for c in stored] == [chunk_id(page.id, n) for n in range(4)]
        assert stored[2].embed_text.startswith("Page 1\nSection 2\n\n")
        assert await repo.count_version(_SOURCE, _V1) == 4

    async def test_saving_the_same_page_again_is_idempotent(self, repo: ChunkRepository) -> None:
        page = _page(1, _markdown(3))

        await repo.save_page_chunks(page.id, _chunks(page))
        await repo.save_page_chunks(page.id, _chunks(page))  # e.g. a resumed ingest

        assert await repo.count({"page_id": page.id}) == 3
        assert [c.id for c in await repo.for_page(page.id)] == [
            chunk_id(page.id, n) for n in range(3)
        ]

    async def test_rechunking_replaces_the_page_and_drops_a_stale_tail(
        self, repo: ChunkRepository
    ) -> None:
        page = _page(1, _markdown(5))
        await repo.save_page_chunks(page.id, _chunks(page))

        fewer = _chunks(page, _markdown(2).replace("widgets", "parcels"))
        await repo.save_page_chunks(page.id, fewer)

        stored = await repo.for_page(page.id)
        assert [c.ordinal for c in stored] == [0, 1]
        assert all("parcels" in c.text for c in stored)

    async def test_an_empty_chunking_removes_the_pages_chunks(self, repo: ChunkRepository) -> None:
        page, other = _page(1, _markdown(3)), _page(2, _markdown(2))
        await repo.save_page_chunks(page.id, _chunks(page))
        await repo.save_page_chunks(other.id, _chunks(other))

        assert await repo.save_page_chunks(page.id, []) == 0

        assert await repo.for_page(page.id) == []
        assert len(await repo.for_page(other.id)) == 2

    async def test_a_version_streams_page_by_page_for_re_embedding(
        self, repo: ChunkRepository
    ) -> None:
        pages = [_page(n, _markdown(n + 1)) for n in range(4)]
        for page in reversed(pages):  # saved out of order
            await repo.save_page_chunks(page.id, _chunks(page))
        newer = _page(9, _markdown(2), version=_V2)
        await repo.save_page_chunks(newer.id, _chunks(newer))

        streamed = [c async for c in repo.iter_version(_SOURCE, _V1, batch_size=2)]

        expected = sorted((c.page_id, c.ordinal) for p in pages for c in _chunks(p))
        assert [(c.page_id, c.ordinal) for c in streamed] == expected
        assert {c.version for c in streamed} == {_V1}
        assert all(c.embed_text and c.token_count > 0 for c in streamed)  # ready to embed
        assert await repo.count_version(_SOURCE, _V2) == 2

    async def test_cleanup_by_version_and_by_source(self, repo: ChunkRepository) -> None:
        for page in (
            _page(1, _markdown(2)),
            _page(2, _markdown(3), version=_V2),
            _page(3, _markdown(1), source=_OTHER_SOURCE),
        ):
            await repo.save_page_chunks(page.id, _chunks(page))

        assert await repo.versions(_SOURCE) == [_V1, _V2]
        assert await repo.delete_version(_SOURCE, _V1) == 2
        assert await repo.versions(_SOURCE) == [_V2]
        assert await repo.delete_source(_SOURCE) == 3
        assert await repo.versions(_SOURCE) == []
        assert await repo.count_version(_OTHER_SOURCE, _V1) == 1  # another source: untouched

    async def test_invalid_page_writes_are_refused(self, repo: ChunkRepository) -> None:
        page, other = _page(1, _markdown(2)), _page(2, _markdown(2))
        chunks = _chunks(page)

        with pytest.raises(ValueError, match="belong to the page"):
            await repo.save_page_chunks(page.id, [*chunks, *_chunks(other)])
        with pytest.raises(ValueError, match="ordinals"):
            await repo.save_page_chunks(page.id, chunks[1:])
        with pytest.raises(ValueError, match="chunk_id"):
            await repo.save_page_chunks(page.id, [c.model_copy(update={"id": "chk_" + "f" * 32})
                                                  for c in chunks])  # fmt: skip
        with pytest.raises(ValueError, match="one source version"):
            await repo.save_page_chunks(
                page.id, [chunks[0], chunks[1].model_copy(update={"version": _V2})]
            )
        assert await repo.for_page(page.id) == []  # nothing was written

    async def test_indexes(self, repo: ChunkRepository) -> None:
        indexes = await repo.collection.index_information()

        assert list(indexes["source_version_order"]["key"]) == [
            ("source_id", 1),
            ("version", 1),
            ("page_id", 1),
            ("ordinal", 1),
        ]
        assert indexes["page_ordinal_unique"]["unique"] is True

    def test_chunk_ids_are_deterministic_per_page_and_ordinal(self) -> None:
        assert chunk_id("page_a", 0) == chunk_id("page_a", 0)
        assert len({chunk_id("page_a", 0), chunk_id("page_a", 1), chunk_id("page_b", 0)}) == 3
        assert chunk_id("page_a", 0).startswith("chk_") and len(chunk_id("page_a", 0)) == 36
