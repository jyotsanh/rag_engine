"""Async Qdrant client lifecycle.

Only the connection lives here. Collections, upserts and search belong to the
`vectorstores` package (Steps 67-74), which receives this client.
"""

from qdrant_client import AsyncQdrantClient

from rag_engine.config import Settings, logger, settings
from rag_engine.infrastructure.exceptions import VectorStoreUnavailableError


class QdrantConnection:
    """Owns one `AsyncQdrantClient` per process."""

    def __init__(self, config: Settings = settings) -> None:
        self._config = config
        self._client: AsyncQdrantClient | None = None

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise VectorStoreUnavailableError("Qdrant connection is not open")
        return self._client

    async def connect(self) -> None:
        if self._client is not None:
            return
        api_key = self._config.QDRANT_API_KEY
        self._client = AsyncQdrantClient(
            host=self._config.QDRANT_HOST,
            port=self._config.QDRANT_HTTP_PORT,
            grpc_port=self._config.QDRANT_GRPC_PORT,
            prefer_grpc=self._config.QDRANT_PREFER_GRPC,
            https=self._config.QDRANT_HTTPS,
            api_key=api_key.get_secret_value() if api_key else None,
            timeout=self._config.QDRANT_TIMEOUT_SECONDS,
            # skip the blocking version handshake at construction; ping() checks reachability
            check_compatibility=False,
        )
        try:
            await self.ping()
        except VectorStoreUnavailableError:
            await self.close()
            raise
        logger.info("connected to Qdrant at {}", self._config.qdrant_url)

    async def ping(self) -> None:
        """An authenticated call, so a wrong API key fails here and not at first use."""
        try:
            await self.client.get_collections()
        except Exception as exc:  # qdrant-client raises transport-specific errors (httpx, grpc)
            raise VectorStoreUnavailableError(details={"reason": type(exc).__name__}) from exc

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
