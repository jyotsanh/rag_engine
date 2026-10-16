"""Session-scoped MongoDB / Redis / Qdrant containers for integration tests.

Containers start lazily (only when a test requests them) and once per session.
`integration_settings` points the global `settings` at them for one test and
gives the test an isolated Mongo database, a flushed Redis DB and an empty Qdrant.
If Docker is unavailable, every integration test is skipped, not failed.
"""

import secrets
import uuid
from collections.abc import Iterator

import docker
import pytest
from pymongo import MongoClient
from qdrant_client import QdrantClient
from redis import Redis
from testcontainers.community.mongodb import MongoDbContainer
from testcontainers.community.qdrant import QdrantContainer
from testcontainers.community.redis import RedisContainer

from rag_engine.config import Settings
from tests.conftest import OverrideSettings
from tests.support.services import (
    MONGO_IMAGE,
    QDRANT_IMAGE,
    REDIS_IMAGE,
    MongoService,
    QdrantService,
    RedisService,
)


@pytest.fixture(scope="session")
def docker_available() -> None:
    try:
        docker.from_env().ping()
    except Exception as exc:  # any failure means "no usable docker daemon"
        pytest.skip(f"docker unavailable: {exc}")


@pytest.fixture(scope="session")
def mongo_service(docker_available: None) -> Iterator[MongoService]:
    container = MongoDbContainer(MONGO_IMAGE, username="rag", password=secrets.token_hex(12))
    with container:
        yield MongoService(
            host=container.get_container_host_ip(),
            port=int(container.get_exposed_port(27017)),
            username=container.username,
            password=container.password,
        )


@pytest.fixture(scope="session")
def redis_service(docker_available: None) -> Iterator[RedisService]:
    password = secrets.token_hex(12)
    with RedisContainer(REDIS_IMAGE, password=password) as container:
        yield RedisService(
            host=container.get_container_host_ip(),
            port=int(container.get_exposed_port(6379)),
            password=password,
        )


@pytest.fixture(scope="session")
def qdrant_service(docker_available: None) -> Iterator[QdrantService]:
    api_key = secrets.token_hex(12)
    with QdrantContainer(QDRANT_IMAGE, api_key=api_key) as container:
        yield QdrantService(
            host=container.get_container_host_ip(),
            http_port=int(container.get_exposed_port(6333)),
            grpc_port=int(container.get_exposed_port(6334)),
            api_key=api_key,
        )


@pytest.fixture
def integration_settings(
    override_settings: OverrideSettings,
    mongo_service: MongoService,
    redis_service: RedisService,
    qdrant_service: QdrantService,
) -> Iterator[Settings]:
    """Global settings aimed at the containers, with per-test data isolation."""
    db_name = f"test_{uuid.uuid4().hex[:12]}"
    configured = override_settings(
        MONGO_URI=mongo_service.url,
        MONGO_DB=db_name,
        REDIS_HOST=redis_service.host,
        REDIS_PORT=redis_service.port,
        REDIS_PASSWORD=redis_service.password,
        QDRANT_HOST=qdrant_service.host,
        QDRANT_HTTP_PORT=qdrant_service.http_port,
        QDRANT_GRPC_PORT=qdrant_service.grpc_port,
        QDRANT_API_KEY=qdrant_service.api_key,
        QDRANT_HTTPS=False,
    )
    yield configured
    _reset_services(db_name, mongo_service, redis_service, qdrant_service, configured)


def _reset_services(
    db_name: str,
    mongo: MongoService,
    redis: RedisService,
    qdrant: QdrantService,
    configured: Settings,
) -> None:
    with MongoClient(mongo.url) as client:
        client.drop_database(db_name)
    redis_dbs = {
        configured.REDIS_DB,
        configured.REDIS_CELERY_BROKER_DB,
        configured.REDIS_CELERY_RESULT_DB,
    }
    for db in redis_dbs:
        with Redis(host=redis.host, port=redis.port, password=redis.password, db=db) as client:
            client.flushdb()
    client = QdrantClient(url=qdrant.url, api_key=qdrant.api_key, https=False)
    try:
        for collection in client.get_collections().collections:
            client.delete_collection(collection.name)
    finally:
        client.close()
