"""In-memory stand-ins for infrastructure, for tests that don't need real services."""

import hashlib
import math
import random
from collections.abc import Callable, Mapping, Sequence

from rag_engine.config import EmbeddingModelName, EmbeddingProvider, VectorStoreType
from rag_engine.crawling import CachedRobots
from rag_engine.domain import ContentHashClaim
from rag_engine.embeddings import (
    BaseEmbedder,
    EmbeddingUsageCounts,
    InputType,
    ProviderResult,
    SparseVector,
    Vector,
)
from rag_engine.infrastructure import DatabaseUnavailableError, HealthCheck
from rag_engine.vectorstores import (
    BaseVectorStore,
    CollectionMismatchError,
    CollectionNotFoundError,
    CollectionSpec,
    SearchHit,
    VectorFilter,
    VectorPoint,
)


class NoopInfrastructure:
    """Satisfies `Infrastructure`; records lifecycle calls and serves the given health checks."""

    def __init__(self, checks: Mapping[str, HealthCheck] | None = None) -> None:
        self.started = False
        self.stopped = False
        self.checks: dict[str, HealthCheck] = dict(checks or {})

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def health_checks(self) -> Mapping[str, HealthCheck]:
        return self.checks


class InMemoryRobotsStore:
    """Satisfies `RobotsStore` without Redis. Records the TTL of every write."""

    def __init__(self) -> None:
        self.current: dict[str, CachedRobots] = {}
        self.last_good: dict[str, CachedRobots] = {}
        self.ttls: dict[tuple[str, str], float] = {}

    async def load(self, origin: str) -> CachedRobots | None:
        return self.current.get(origin)

    async def save(self, origin: str, entry: CachedRobots, ttl: float) -> None:
        self.current[origin] = entry
        self.ttls[("current", origin)] = ttl

    async def load_last_good(self, origin: str) -> CachedRobots | None:
        return self.last_good.get(origin)

    async def save_last_good(self, origin: str, entry: CachedRobots, ttl: float) -> None:
        self.last_good[origin] = entry
        self.ttls[("last_good", origin)] = ttl


class InMemoryContentClaimStore:
    """Satisfies `ContentClaimStore`. No await between lookup and insert: atomic on one loop."""

    def __init__(self) -> None:
        self.claims: dict[tuple[str, str], ContentHashClaim] = {}

    async def claim(
        self, job_id: str, content_hash: str, page_id: str, url: str
    ) -> ContentHashClaim:
        key = (job_id, content_hash)
        if key not in self.claims:
            self.claims[key] = ContentHashClaim(
                job_id=job_id, content_hash=content_hash, page_id=page_id, url=url
            )
        return self.claims[key]


def fake_vector(text: str, dimensions: int) -> Vector:
    """A unit vector determined by `text` alone: the same text always embeds the same."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    values = [rng.gauss(0.0, 1.0) for _ in range(dimensions)]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


class FakeEmbedder(BaseEmbedder):
    """A `BaseEmbedder` with deterministic vectors and no network (for OpenAI-model spaces).

    Records every provider call; `respond` replaces the provider call (to return a broken
    response or raise a provider error). Usage counts one request per call and one token
    per whitespace-separated word.
    """

    provider = EmbeddingProvider.OPENAI

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # ty: ignore[invalid-argument-type]
        self.calls: list[tuple[list[str], InputType]] = []
        self.respond: Callable[[list[str], InputType], ProviderResult] | None = None
        self.closed = False

    async def _embed(self, texts: list[str], input_type: InputType) -> ProviderResult:
        self.calls.append((list(texts), input_type))
        if self.respond is not None:
            return self.respond(texts, input_type)
        return ProviderResult(
            vectors=[fake_vector(text, self.dimensions) for text in texts],
            usage=EmbeddingUsageCounts(requests=1, tokens=sum(len(t.split()) for t in texts)),
        )

    async def aclose(self) -> None:
        self.closed = True


class InMemoryEmbeddingCache:
    """Satisfies `EmbeddingCacheStore`; `fail_reads`/`fail_writes` simulate a Mongo outage."""

    def __init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}
        self.reads: list[list[str]] = []
        self.fail_reads = False
        self.fail_writes = False

    async def get_many(self, keys: Sequence[str], *, dimensions: int) -> dict[str, list[float]]:
        self.reads.append(list(keys))
        if self.fail_reads:
            raise DatabaseUnavailableError(details={"reason": "ServerSelectionTimeoutError"})
        return {k: v for k in keys if (v := self.vectors.get(k)) and len(v) == dimensions}

    async def put_many(
        self, vectors: Mapping[str, list[float]], *, model: EmbeddingModelName, dimensions: int
    ) -> int:
        if self.fail_writes:
            raise DatabaseUnavailableError(details={"reason": "AutoReconnect"})
        self.vectors.update(vectors)
        return len(vectors)


RRF_K = 60  # reciprocal rank fusion constant (the usual default)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norms if norms else 0.0


class InMemoryVectorStore(BaseVectorStore):
    """A reference `BaseVectorStore`: exact cosine, IDF-weighted sparse dot product, RRF.

    Slow (linear scans) and process-local; it defines the semantics a real backend must
    reproduce, so pipeline and search tests can run without Qdrant.
    """

    kind = VectorStoreType.QDRANT  # stands in for the active backend in tests

    def __init__(self) -> None:
        self.collections: dict[str, tuple[CollectionSpec, dict[str, VectorPoint]]] = {}
        self.closed = False

    def _points(self, spec: CollectionSpec) -> dict[str, VectorPoint]:
        if spec.name not in self.collections:
            raise CollectionNotFoundError(details={"collection": spec.name})
        return self.collections[spec.name][1]

    def _matching(self, spec: CollectionSpec, filter_: VectorFilter) -> list[VectorPoint]:
        return [p for p in self._points(spec).values() if filter_.matches(p.payload)]

    async def _ensure_collection(self, spec: CollectionSpec) -> bool:
        existing = self.collections.get(spec.name)
        if existing is None:
            self.collections[spec.name] = (spec, {})
            return True
        if existing[0] != spec:
            raise CollectionMismatchError(details={"collection": spec.name})
        return False

    async def _upsert(self, spec: CollectionSpec, points: list[VectorPoint]) -> None:
        stored = self._points(spec)
        for point in points:
            stored[point.id] = point

    async def _delete_by_filter(self, spec: CollectionSpec, filter_: VectorFilter) -> None:
        stored = self._points(spec)
        for point in self._matching(spec, filter_):
            del stored[point.id]

    async def _count(self, spec: CollectionSpec, filter_: VectorFilter) -> int:
        return len(self._matching(spec, filter_))

    async def _search_dense(
        self,
        spec: CollectionSpec,
        vector: list[float],
        filter_: VectorFilter,
        limit: int,
        score_threshold: float | None,
    ) -> list[SearchHit]:
        hits = [
            SearchHit(p.id, _cosine(vector, p.dense), p.payload)
            for p in self._matching(spec, filter_)
        ]
        if score_threshold is not None:
            hits = [h for h in hits if h.score >= score_threshold]
        return sorted(hits, key=lambda h: (-h.score, h.id))[:limit]

    def _sparse_ranking(self, points: list[VectorPoint], query: SparseVector) -> list[VectorPoint]:
        """Points by IDF-weighted dot product (IDF over the filtered points, like Qdrant)."""
        total = len(points)
        frequency: dict[int, int] = {}
        for point in points:
            for index in point.sparse.indices if point.sparse else ():
                frequency[index] = frequency.get(index, 0) + 1
        weights = dict(zip(query.indices, query.values, strict=True))
        scored = []
        for point in points:
            if point.sparse is None:
                continue
            score = sum(
                weights[i] * v * math.log(1 + (total - frequency[i] + 0.5) / (frequency[i] + 0.5))
                for i, v in zip(point.sparse.indices, point.sparse.values, strict=True)
                if i in weights
            )
            if score > 0:
                scored.append((score, point))
        return [p for _, p in sorted(scored, key=lambda sp: (-sp[0], sp[1].id))]

    async def _search_hybrid(
        self,
        spec: CollectionSpec,
        dense: list[float],
        sparse: SparseVector,
        filter_: VectorFilter,
        limit: int,
        prefetch_limit: int,
    ) -> list[SearchHit]:
        points = self._matching(spec, filter_)
        by_dense = sorted(points, key=lambda p: (-_cosine(dense, p.dense), p.id))[:prefetch_limit]
        by_sparse = self._sparse_ranking(points, sparse)[:prefetch_limit]
        fused: dict[str, float] = {}
        for ranking in (by_dense, by_sparse):
            for rank, point in enumerate(ranking, start=1):
                fused[point.id] = fused.get(point.id, 0.0) + 1.0 / (RRF_K + rank)
        payloads = {p.id: p.payload for p in points}
        ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return [SearchHit(pid, score, payloads[pid]) for pid, score in ordered]

    async def aclose(self) -> None:
        self.closed = True
