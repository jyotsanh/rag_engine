"""In-memory stand-ins for infrastructure, for tests that don't need real services."""

from collections.abc import Mapping

from rag_engine.crawling import CachedRobots
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
