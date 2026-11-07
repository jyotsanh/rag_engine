"""Session-scoped MongoDB / Redis / Qdrant containers (a pytest plugin, see `tests/conftest.py`).

Registered for the whole test tree so integration tests *and* browser tests that need
real services (the crawl E2E, Step 54) share one set of containers.
Containers start lazily (only when a test requests them) and once per session.
`integration_settings` points the global `settings` at them for one test and
gives the test an isolated Mongo database, a flushed Redis DB and an empty Qdrant.
If Docker is unavailable, every test that needs a container is skipped, not failed.
"""

import secrets
import uuid
from collections.abc import Iterator

import docker
import docker.types
import pytest
from pymongo import MongoClient
from qdrant_client import QdrantClient
from redis import Redis
from testcontainers.community.mongodb import MongoDbContainer
from testcontainers.community.qdrant import QdrantContainer
from testcontainers.community.redis import RedisContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

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
    # WiredTiger opens a file per collection and index, and every test gets its own
    # database: the Docker default of 1024 runs out (EMFILE → mongod aborts)
    container.with_kwargs(ulimits=[docker.types.Ulimit(name="nofile", soft=64000, hard=64000)])
    # the image starts a temporary mongod to create the root user, then restarts it:
    # "Waiting for connections" is logged twice and only the second one is the real server
    container.waiting_for(LogMessageWaitStrategy("Waiting for connections", times=2))
    with container:
        service = MongoService(
            host=container.get_container_host_ip(),
            port=int(container.get_exposed_port(27017)),
            username=container.username,
            password=container.password,
        )
        with MongoClient(service.url, serverSelectionTimeoutMS=20_000) as client:
            client.admin.command("ping")  # authenticated round-trip before any test runs
        yield service


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
    # short timeouts: if a service died mid-run, fail fast instead of 30s per test
    with MongoClient(mongo.url, serverSelectionTimeoutMS=2_000) as client:
        client.drop_database(db_name)
    redis_dbs = {
        configured.REDIS_DB,
        configured.REDIS_CELERY_BROKER_DB,
        configured.REDIS_CELERY_RESULT_DB,
    }
    for db in redis_dbs:
        with Redis(
            host=redis.host,
            port=redis.port,
            password=redis.password,
            db=db,
            socket_connect_timeout=2,
            socket_timeout=5,
        ) as client:
            client.flushdb()
    client = QdrantClient(url=qdrant.url, api_key=qdrant.api_key, https=False, timeout=5)
    try:
        for collection in client.get_collections().collections:
            client.delete_collection(collection.name)
    finally:
        client.close()
