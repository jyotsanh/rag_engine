"""Tenant API key auth against real MongoDB, through the real app."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import APIRouter, FastAPI

from rag_engine.api import create_app
from rag_engine.api.deps import CurrentTenant, get_api_key_hasher
from rag_engine.config import Settings, get_log_context
from rag_engine.domain import Tenant, TenantStatus, utc_now
from rag_engine.infrastructure import MongoConnection
from rag_engine.repositories import ApiKeyRepository, TenantRepository
from rag_engine.security import ApiKeyHasher
from tests.conftest import OverrideSettings

OLD_PEPPER = "old-pepper-0123456789abcdef0123456789abcdef"


class _SpyHasher(ApiKeyHasher):
    def __init__(self, config: Settings) -> None:
        super().__init__(config)
        self.burns = 0

    def burn(self, plaintext=None) -> None:
        self.burns += 1
        super().burn(plaintext)


@dataclass
class _World:
    client: httpx.AsyncClient
    app: FastAPI
    tenants: TenantRepository
    keys: ApiKeyRepository
    hasher: _SpyHasher
    tenant: Tenant

    async def new_key(self, **kwargs) -> tuple[str, str]:
        record, generated = await self.keys.issue(self.tenant.id, self.hasher, **kwargs)
        return record.id, generated.plaintext.get_secret_value()

    async def call(self, key: str | None) -> httpx.Response:
        headers = {"Authorization": f"Bearer {key}"} if key is not None else {}
        return await self.client.get("/_test/whoami", headers=headers)


@pytest.fixture
async def world(integration_settings: Settings) -> AsyncIterator[_World]:
    app = create_app()
    router = APIRouter()

    @router.get("/_test/whoami")
    async def whoami(ctx: CurrentTenant) -> dict[str, str | None]:
        return {"tenant_id": ctx.tenant_id, "logged_tenant": get_log_context()["tenant_id"]}

    app.include_router(router)
    hasher = _SpyHasher(integration_settings)
    app.dependency_overrides[get_api_key_hasher] = lambda: hasher

    mongo = MongoConnection(integration_settings)
    await mongo.connect()
    transport = httpx.ASGITransport(app=app)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
    ):
        tenants = TenantRepository(mongo.db)
        tenant = await tenants.create(Tenant(name="Acme"))
        yield _World(client, app, tenants, ApiKeyRepository(mongo.db), hasher, tenant)
    await mongo.close()


def _assert_uniform_401(response: httpx.Response) -> None:
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    error = response.json()["error"]
    assert (error["code"], error["message"], error["details"]) == (
        "invalid_credentials",
        "Missing or invalid API key",
        {},
    )


# ---- #
# Tests
# ---- #


class TestTenantAuth_S17:
    """S17 — Auth dependencies: tenant API key → TenantContext, uniform 401, disabled → 403."""

    async def test_valid_key_resolves_tenant_and_binds_log_context(self, world: _World) -> None:
        _, key = await world.new_key()

        response = await world.call(key)

        assert response.status_code == 200
        assert response.json() == {"tenant_id": world.tenant.id, "logged_tenant": world.tenant.id}

    async def test_missing_and_malformed_keys(self, world: _World) -> None:
        for key in (None, "", "not-a-key", "rk_short_x"):
            _assert_uniform_401(await world.call(key))

    async def test_unknown_prefix_burns_hash_work(self, world: _World) -> None:
        response = await world.call("rk_NoSuchPrefix_" + "x" * 43)

        _assert_uniform_401(response)
        assert world.hasher.burns == 1

    async def test_wrong_secret(self, world: _World) -> None:
        _, key = await world.new_key()
        wrong = key[:-4] + ("AAAA" if not key.endswith("AAAA") else "BBBB")

        _assert_uniform_401(await world.call(wrong))

    async def test_revoked_key(self, world: _World) -> None:
        key_id, key = await world.new_key()
        await world.keys.revoke(key_id, world.tenant.id)

        _assert_uniform_401(await world.call(key))

    async def test_expired_key(self, world: _World) -> None:
        key_id, key = await world.new_key(expires_at=utc_now() + timedelta(minutes=5))
        # a realistic expired key: created a while ago, expired a moment ago
        now = utc_now()
        await world.keys.collection.update_one(
            {"_id": key_id},
            {
                "$set": {
                    "created_at": now - timedelta(hours=1),
                    "expires_at": now - timedelta(seconds=1),
                }
            },
        )

        _assert_uniform_401(await world.call(key))

    async def test_deleted_tenant(self, world: _World) -> None:
        _, key = await world.new_key()
        await world.tenants.delete(world.tenant.id)

        _assert_uniform_401(await world.call(key))

    async def test_disabled_tenant_is_403(self, world: _World) -> None:
        _, key = await world.new_key()
        await world.tenants.set_status(world.tenant.id, TenantStatus.DISABLED)

        response = await world.call(key)

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "tenant_disabled"

    async def test_successful_use_updates_last_used(self, world: _World) -> None:
        key_id, key = await world.new_key()

        await world.call(key)

        stored = await world.keys.get_or_raise(key_id)
        assert stored.last_used_at is not None

    async def test_key_from_retired_pepper_is_accepted_and_rehashed(
        self, world: _World, override_settings: OverrideSettings
    ) -> None:
        old = ApiKeyHasher(override_settings(API_KEY_PEPPER=OLD_PEPPER))
        record, generated = await world.keys.issue(world.tenant.id, old)
        rotated = _SpyHasher(
            override_settings(
                API_KEY_PEPPER="new-pepper-0123456789abcdef0123456789abcdef",
                API_KEY_PEPPER_PREVIOUS=[OLD_PEPPER],
            )
        )
        world.app.dependency_overrides[get_api_key_hasher] = lambda: rotated

        response = await world.call(generated.plaintext.get_secret_value())

        assert response.status_code == 200
        stored = await world.keys.get_or_raise(record.id)
        assert stored.key_hash != record.key_hash
        assert rotated.verify(generated.plaintext, stored.key_hash).needs_rehash is False

    async def test_admin_key_is_not_a_tenant_key(
        self, world: _World, integration_settings: Settings
    ) -> None:
        _assert_uniform_401(await world.call(integration_settings.ADMIN_API_KEY.get_secret_value()))
