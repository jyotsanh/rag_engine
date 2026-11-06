"""Connection details for the backing services integration tests run against."""

from dataclasses import dataclass
from urllib.parse import quote

# keep in lockstep with docker-compose.yml so tests exercise the deployed versions
MONGO_IMAGE = "mongo:8.0"
REDIS_IMAGE = "redis:8.2"
QDRANT_IMAGE = "qdrant/qdrant:v1.19.2-unprivileged"


@dataclass(frozen=True, slots=True)
class MongoService:
    host: str
    port: int
    username: str
    password: str

    @property
    def url(self) -> str:
        return f"mongodb://{quote(self.username)}:{quote(self.password)}@{self.host}:{self.port}"


@dataclass(frozen=True, slots=True)
class RedisService:
    host: str
    port: int
    password: str

    def url(self, db: int = 0) -> str:
        return f"redis://:{quote(self.password)}@{self.host}:{self.port}/{db}"


@dataclass(frozen=True, slots=True)
class QdrantService:
    host: str
    http_port: int
    grpc_port: int
    api_key: str

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.http_port}"
