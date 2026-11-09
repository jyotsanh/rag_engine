"""Version swap semantics: Mongo `active_version` flip + Qdrant version filter, end to end."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from rag_engine.config import Settings
from rag_engine.domain import Source, SourceStats, utc_now
from rag_engine.embeddings import SparseVector
from rag_engine.infrastructure import MongoConnection, QdrantConnection
from rag_engine.repositories import (
    SourceNotFoundError,
    SourceRepository,
    SourceVersionConflictError,
)
from rag_engine.services import search_scope
from rag_engine.vectorstores import (
    ChunkPayload,
    CollectionSpec,
    QdrantVectorStore,
    VectorFilter,
    VectorPoint,
    point_id,
)
from tests.support.fakes import fake_vector

_DIMS = 16
_SPEC = CollectionSpec("version_swap_test", _DIMS)
_TENANT, _OTHER_TENANT = "ten_" + "1" * 32, "ten_" + "2" * 32
_KB, _OTHER_KB = "kb_" + "1" * 32, "kb_" + "2" * 32


def _version(n: int) -> str:
    return f"job_{n:032x}"


def _points(source_id: str, version: str, count: int, *, tenant: str = _TENANT,
            kb: str = _KB) -> list[VectorPoint]:  # fmt: skip
    """`count` chunks of one page; texts are the same in every version (a re-crawl)."""
    points = []
    for n in range(count):
        payload = ChunkPayload(
            tenant_id=tenant, kb_id=kb, source_id=source_id, version=version,
            chunk_id=f"chk_{n:032x}", page_id=f"page_{source_id[-4:]}_{version[-4:]}",
            url=f"https://acme.example/{n}", ordinal=n, text=f"widget fact number {n}",
        )  # fmt: skip
        points.append(VectorPoint(point_id(version, payload.chunk_id),
                                  fake_vector(payload.text, _DIMS), payload,
                                  SparseVector((n,), (1.0,))))  # fmt: skip
    return points


@dataclass
class _Env:
    sources: SourceRepository
    store: QdrantVectorStore

    async def source(self, *, tenant: str = _TENANT, kb: str = _KB, seed: str = "a") -> Source:
        source, _ = await self.sources.get_or_create(tenant, kb, f"https://{seed}.example/")
        return source

    async def search(self, tenant: str = _TENANT, kb: str = _KB, **scope: object) -> set[str]:
        """The versions the hits of a KB search come from (empty when nothing is servable)."""
        where = await search_scope(self.sources, tenant, kb, **scope)  # ty: ignore[invalid-argument-type]
        if where is None:
            return set()
        hits = await self.store.search_dense(_SPEC, fake_vector("widget fact number 3", _DIMS),
                                             where, limit=50)  # fmt: skip
        return {h.payload.version for h in hits}


@pytest.fixture
async def env(integration_settings: Settings) -> AsyncIterator[_Env]:
    mongo, qdrant = MongoConnection(integration_settings), QdrantConnection(integration_settings)
    await mongo.connect()
    await mongo.ensure_indexes([Source])
    await qdrant.connect()
    store = QdrantVectorStore(qdrant.client)
    await store.ensure_collection(_SPEC)
    yield _Env(SourceRepository(mongo.db), store)
    await qdrant.close()
    await mongo.close()


def _stats(chunks: int) -> SourceStats:
    return SourceStats(pages=1, chunks=chunks, last_succeeded_at=utc_now())


# ---- #
# Tests
# ---- #


class TestVersionSwap_S73:
    """S73 — Version swap: search serves the old version until the atomic flip, then the new."""

    async def test_search_serves_the_old_version_during_a_concurrent_reingest(
        self, env: _Env
    ) -> None:
        source = await env.source()
        v1, v2 = _version(1), _version(2)
        await env.store.upsert(_SPEC, _points(source.id, v1, 20))
        assert await env.sources.activate_version(source.id, v1, stats=_stats(20)) is None

        new_points = _points(source.id, v2, 40)
        seen_during_ingest: list[set[str]] = []
        ingesting = True

        async def reingest() -> None:
            nonlocal ingesting
            for start in range(0, len(new_points), 5):  # v2 arrives in 8 batches
                await env.store.upsert(_SPEC, new_points[start : start + 5])
                await asyncio.sleep(0)
            ingesting = False

        async def search_continuously() -> None:
            while ingesting:
                seen_during_ingest.append(await env.search())

        await asyncio.gather(reingest(), search_continuously())
        before_flip = await env.search()
        previous = await env.sources.activate_version(source.id, v2, stats=_stats(40))
        after_flip = await env.search()

        assert len(seen_during_ingest) >= 3  # searches really overlapped the ingest
        assert all(versions == {v1} for versions in seen_during_ingest)
        assert before_flip == {v1}  # fully ingested but not flipped: still invisible
        assert previous == v1  # handed back for cleanup (Step 78)
        assert after_flip == {v2}  # the flip switches search at once
        assert await env.store.count(_SPEC, VectorFilter(_TENANT, versions=(v1,))) == 20
        stored = await env.sources.get_or_raise(source.id)
        assert (stored.active_version, stored.stats.chunks) == (v2, 40)

    async def test_a_kb_serves_each_sources_own_active_version(self, env: _Env) -> None:
        served, other_served = await env.source(seed="a"), await env.source(seed="b")
        first_crawl, deleted = await env.source(seed="c"), await env.source(seed="d")
        for source, version in ((served, 1), (other_served, 3), (deleted, 5)):
            await env.store.upsert(_SPEC, _points(source.id, _version(version), 4))
            await env.sources.activate_version(source.id, _version(version), stats=_stats(4))
        await env.store.upsert(_SPEC, _points(first_crawl.id, _version(4), 4))  # not yet active
        await env.sources.mark_deleting(deleted.id, _TENANT, _KB)

        everything = await env.search()
        only_b = await env.search(source_ids=(other_served.id,))

        assert everything == {_version(1), _version(3)}  # not the in-progress or deleted one
        assert only_b == {_version(3)}
        assert await env.sources.active_versions(_TENANT, _KB) == {
            served.id: _version(1),
            other_served.id: _version(3),
        }

    async def test_nothing_is_searched_before_any_version_is_active(self, env: _Env) -> None:
        source = await env.source()
        await env.store.upsert(_SPEC, _points(source.id, _version(1), 4))  # first crawl running

        scope = await search_scope(env.sources, _TENANT, _KB)

        assert scope is None  # never a version-less filter that would serve these points
        assert await env.search() == set()

    async def test_active_versions_are_tenant_and_kb_scoped(self, env: _Env) -> None:
        their_kb = "kb_" + "3" * 32  # KB ids are global: another tenant has its own KB
        mine = await env.source()
        other_kb = await env.source(kb=_OTHER_KB)
        theirs = await env.source(tenant=_OTHER_TENANT, kb=their_kb)
        for source in (mine, other_kb, theirs):
            await env.sources.activate_version(source.id, _version(7), stats=_stats(1))

        assert await env.sources.active_versions(_TENANT, _KB) == {mine.id: _version(7)}
        assert await env.sources.active_versions(_TENANT, _OTHER_KB) == {other_kb.id: _version(7)}
        assert list(await env.sources.active_versions(_OTHER_TENANT, their_kb)) == [theirs.id]
        assert await env.sources.active_versions(_OTHER_TENANT, _KB) == {}  # wrong tenant
        assert await env.sources.active_versions(_TENANT, their_kb) == {}

    async def test_compare_and_swap_refuses_a_stale_flip(self, env: _Env) -> None:
        source = await env.source()
        flip = env.sources.activate_version
        await flip(source.id, _version(1), stats=_stats(1))
        await flip(source.id, _version(2), stats=_stats(1), check_previous=True,
                   expected_previous=_version(1))  # fmt: skip

        with pytest.raises(SourceVersionConflictError) as exc:
            await flip(source.id, _version(3), stats=_stats(1), check_previous=True,
                       expected_previous=_version(1))  # fmt: skip

        assert exc.value.details["active_version"] == _version(2)
        assert (await env.sources.get_or_raise(source.id)).active_version == _version(2)

    async def test_a_source_being_deleted_cannot_be_flipped(self, env: _Env) -> None:
        source = await env.source()
        await env.sources.mark_deleting(source.id, _TENANT, _KB)

        with pytest.raises(SourceVersionConflictError) as exc:
            await env.sources.activate_version(source.id, _version(1), stats=_stats(1))
        with pytest.raises(SourceNotFoundError):
            await env.sources.activate_version("src_" + "9" * 32, _version(1), stats=_stats(1))

        assert exc.value.details["status"] == "deleting"
        assert (await env.sources.get_or_raise(source.id)).active_version is None
