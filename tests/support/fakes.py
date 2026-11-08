"""In-memory stand-ins for infrastructure, for tests that don't need real services."""

import hashlib
import math
import random
from collections.abc import Callable, Mapping

from rag_engine.config import EmbeddingProvider
from rag_engine.crawling import CachedRobots
from rag_engine.domain import ContentHashClaim
from rag_engine.embeddings import (
    BaseEmbedder,
    EmbeddingUsageCounts,
    InputType,
    ProviderResult,
    Vector,
)
from rag_engine.infrastructure import HealthCheck


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
