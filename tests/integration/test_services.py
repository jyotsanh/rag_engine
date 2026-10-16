"""Integration services: containers are reachable, authenticated and isolated per test."""

import httpx
import pytest
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from qdrant_client import QdrantClient, models
from redis import AuthenticationError, Redis
from redis.exceptions import ResponseError

from rag_engine.config import Settings
from tests.support.services import MongoService, QdrantService, RedisService

# ---- #
# Tests
# ---- #


class TestIntegrationServices_S8:
    """S8 — Test infrastructure: testcontainers Mongo/Redis/Qdrant + `integration_settings`."""

    def test_integration_marker_auto_applied(self, request: pytest.FixtureRequest) -> None:
        assert request.node.get_closest_marker("integration") is not None

    def test_mongo_requires_auth(self, mongo_service: MongoService) -> None:
        anonymous = f"mongodb://{mongo_service.host}:{mongo_service.port}"

        with MongoClient(anonymous) as client, pytest.raises(OperationFailure):
            client["x"]["c"].insert_one({"a": 1})
        with MongoClient(mongo_service.url) as client:
            assert client["x"]["c"].insert_one({"a": 1}).acknowledged
            client.drop_database("x")

    def test_redis_requires_auth(self, redis_service: RedisService) -> None:
        with (
            Redis(host=redis_service.host, port=redis_service.port) as anonymous,
            pytest.raises((AuthenticationError, ResponseError)),
        ):
            anonymous.ping()
        with Redis.from_url(redis_service.url()) as client:
            assert client.ping() is True

    def test_qdrant_requires_api_key(self, qdrant_service: QdrantService) -> None:
        anonymous = httpx.get(f"{qdrant_service.url}/collections")
        authed = httpx.get(
            f"{qdrant_service.url}/collections", headers={"api-key": qdrant_service.api_key}
        )
        version = httpx.get(qdrant_service.url, headers={"api-key": qdrant_service.api_key})

        assert anonymous.status_code == 401
        assert authed.status_code == 200
        assert version.json()["version"] == "1.19.2"

    def test_integration_settings_point_at_containers(
        self,
        integration_settings: Settings,
        mongo_service: MongoService,
        redis_service: RedisService,
        qdrant_service: QdrantService,
    ) -> None:
        assert integration_settings.mongo_url == mongo_service.url
        assert integration_settings.MONGO_DB.startswith("test_")
        assert integration_settings.redis_url == redis_service.url(integration_settings.REDIS_DB)
        assert integration_settings.qdrant_url == qdrant_service.url

        with MongoClient(integration_settings.mongo_url) as client:
            assert client.admin.command("ping")["ok"] == 1.0
        with Redis.from_url(integration_settings.redis_url) as client:
            assert client.ping() is True

    def test_writes_data_for_isolation_check(self, integration_settings: Settings) -> None:
        with MongoClient(integration_settings.mongo_url) as client:
            client[integration_settings.MONGO_DB]["probe"].insert_one({"leak": True})
        with Redis.from_url(integration_settings.redis_url) as client:
            client.set("probe", "leak")
        qdrant = QdrantClient(
            url=integration_settings.qdrant_url,
            api_key=integration_settings.QDRANT_API_KEY.get_secret_value()
            if integration_settings.QDRANT_API_KEY
            else None,
            https=False,
        )
        qdrant.create_collection(
            "probe", vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE)
        )
        qdrant.close()

    def test_previous_test_data_was_cleaned_up(
        self,
        integration_settings: Settings,
        mongo_service: MongoService,
        qdrant_service: QdrantService,
    ) -> None:
        # runs after the test above (file order)
        with MongoClient(mongo_service.url) as client:
            leftovers = [n for n in client.list_database_names() if n.startswith("test_")]
            assert leftovers == []  # this test's own DB is created lazily, so none exist yet
        with Redis.from_url(integration_settings.redis_url) as client:
            assert client.get("probe") is None
        qdrant = QdrantClient(url=qdrant_service.url, api_key=qdrant_service.api_key, https=False)
        assert qdrant.get_collections().collections == []
        qdrant.close()
