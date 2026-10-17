"""In-memory stand-ins for infrastructure, for tests that don't need real services."""

from collections.abc import Mapping

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
