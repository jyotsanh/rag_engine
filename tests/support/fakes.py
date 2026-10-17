"""In-memory stand-ins for infrastructure, for tests that don't need real services."""


class NoopInfrastructure:
    """Satisfies `Infrastructure`; records lifecycle calls."""

    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True
