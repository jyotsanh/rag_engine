"""Tenants' webhook signing secrets in real MongoDB: issue / rotate / retire through the
admin API and the service, encrypted at rest, and usable to sign verifiable webhooks."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager
from pydantic import SecretStr

from rag_engine.api import create_app
from rag_engine.config import Settings
from rag_engine.domain import Tenant
from rag_engine.infrastructure import AppInfrastructure, MongoConnection
from rag_engine.repositories import TenantNotFoundError, TenantRepository
from rag_engine.security import SecretBox, SecretUnreadableError
from rag_engine.webhooks import (
    WebhookSecretService,
    sign,
    verify_webhook,
)
from tests.conftest import APP_STARTUP_TIMEOUT

API = "/api/v1/admin"
_BODY = b'{"event":"job.succeeded"}'


@dataclass
class _Env:
    mongo: MongoConnection
    box: SecretBox

    @property
    def tenants(self) -> TenantRepository:
        return TenantRepository(self.mongo.db)

    @property
    def service(self) -> WebhookSecretService:
        return WebhookSecretService(self.tenants, self.box)

    async def tenant(self, name: str = "Acme") -> Tenant:
        return await self.tenants.create(Tenant(name=name))


@pytest.fixture
async def env(integration_settings: Settings) -> AsyncIterator[_Env]:
    mongo = MongoConnection(integration_settings)
    await mongo.connect()
    await mongo.ensure_indexes([Tenant])
    yield _Env(mongo, SecretBox.from_settings(integration_settings))
    await mongo.close()


@pytest.fixture
async def admin(integration_settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(infrastructure_factory=AppInfrastructure)
    headers = {"Authorization": f"Bearer {integration_settings.ADMIN_API_KEY.get_secret_value()}"}
    async with (
        LifespanManager(app, startup_timeout=APP_STARTUP_TIMEOUT),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t", headers=headers
        ) as client,
    ):
        yield client


def _verifies(secrets: list[str], receiver_secret: str) -> bool:
    header = sign(_BODY, secrets, timestamp=1_700_000_000)
    try:
        verify_webhook(_BODY, header, receiver_secret, now=1_700_000_000)
    except Exception:
        return False
    return True


# ---- #
# Tests
# ---- #


class TestWebhookSecrets_S82:
    """S82 — `WebhookSecretService`: per-tenant secrets, sealed at rest, rotation overlap."""

    async def test_issue_seals_the_secret_and_signs_with_it(self, env: _Env) -> None:
        tenant = await env.tenant()

        updated, plaintext = await env.service.rotate(tenant.id)

        raw = await env.mongo.db["tenants"].find_one({"_id": tenant.id})
        assert raw is not None and plaintext not in str(raw)  # never stored in clear
        assert [s.prefix for s in updated.webhook_secrets] == [plaintext[:12]]
        assert await env.service.signing_secrets(updated) == [plaintext]
        assert "sealed" not in repr(updated.webhook_secrets[0])

    async def test_rotation_overlaps_one_previous_secret(self, env: _Env) -> None:
        tenant = await env.tenant()
        _, first = await env.service.rotate(tenant.id)
        _, second = await env.service.rotate(tenant.id)
        rotated, third = await env.service.rotate(tenant.id)

        secrets = await env.service.signing_secrets(rotated)

        assert secrets == [third, second]  # current first; the oldest dropped
        assert _verifies(secrets, second) and _verifies(secrets, third)
        assert not _verifies(secrets, first)

    async def test_drop_previous_ends_the_overlap(self, env: _Env) -> None:
        tenant = await env.tenant()
        await env.service.rotate(tenant.id)
        _, current = await env.service.rotate(tenant.id)

        dropped = await env.service.drop_previous(tenant.id)
        again = await env.service.drop_previous(tenant.id)

        assert await env.service.signing_secrets(dropped) == [current]
        assert again.webhook_secrets == dropped.webhook_secrets  # idempotent

    async def test_a_tenant_without_secrets(self, env: _Env) -> None:
        tenant = await env.tenant()
        # a tenant stored before Step 82 has no field at all
        await env.mongo.db["tenants"].update_one(
            {"_id": tenant.id}, {"$unset": {"webhook_secrets": ""}}
        )
        legacy = await env.tenants.get_or_raise(tenant.id)

        assert await env.service.signing_secrets(legacy) == []
        issued, plaintext = await env.service.rotate(tenant.id)
        assert await env.service.signing_secrets(issued) == [plaintext]

    async def test_concurrent_rotations_never_lose_a_secret(self, env: _Env) -> None:
        tenant = await env.tenant()

        results = await asyncio.gather(
            *(env.service.rotate(tenant.id) for _ in range(3)), return_exceptions=True
        )

        issued = [r[1] for r in results if isinstance(r, tuple)]
        final = await env.service.signing_secrets(await env.tenants.get_or_raise(tenant.id))
        # compare-and-swap: every rotation that returned a secret really stored it, so the
        # final pair are the two most recent of those (none silently overwritten)
        assert len(issued) == 3 and len(final) == 2 and set(final) <= set(issued)

    async def test_an_encryption_key_rotation_reseals_on_use(
        self, env: _Env, integration_settings: Settings
    ) -> None:
        tenant = await env.tenant()
        _, plaintext = await env.service.rotate(tenant.id)
        old_key = integration_settings.SECRET_ENCRYPTION_KEY
        rotated_box = SecretBox([SecretStr("n" * 40), old_key])
        service = WebhookSecretService(env.tenants, rotated_box)

        stored = await env.tenants.get_or_raise(tenant.id)
        assert await service.signing_secrets(stored) == [plaintext]

        resealed = await env.tenants.get_or_raise(tenant.id)
        assert not rotated_box.needs_reseal(resealed.webhook_secrets[0].sealed)
        assert await WebhookSecretService(env.tenants, SecretBox([SecretStr("n" * 40)])) \
            .signing_secrets(resealed) == [plaintext]  # fmt: skip

    async def test_a_secret_from_another_tenant_does_not_open(self, env: _Env) -> None:
        acme, globex = await env.tenant("Acme"), await env.tenant("Globex")
        stolen, _ = await env.service.rotate(acme.id)
        await env.mongo.db["tenants"].update_one(
            {"_id": globex.id},
            {"$set": {"webhook_secrets": [s.model_dump() for s in stolen.webhook_secrets]}},
        )

        with pytest.raises(SecretUnreadableError):
            await env.service.signing_secrets(await env.tenants.get_or_raise(globex.id))

    async def test_an_unknown_tenant_is_404(self, env: _Env) -> None:
        with pytest.raises(TenantNotFoundError):
            await env.service.rotate("ten_" + "f" * 32)


class TestWebhookSecretApi_S82:
    """S82 — `POST /admin/tenants/{id}/webhook-secret` (shown once) and
    `DELETE …/webhook-secret/previous`."""

    async def _tenant(self, admin: httpx.AsyncClient) -> dict[str, Any]:
        response = await admin.post(f"{API}/tenants", json={"name": "Acme"})
        assert response.status_code == 201
        return response.json()

    async def test_issue_rotate_and_retire(self, admin: httpx.AsyncClient) -> None:
        tenant = await self._tenant(admin)
        url = f"{API}/tenants/{tenant['id']}/webhook-secret"

        first = await admin.post(url)
        second = await admin.post(url)
        shown = (await admin.get(f"{API}/tenants/{tenant['id']}")).json()
        retired = await admin.delete(f"{url}/previous")

        assert tenant["webhook_secrets"] == []
        assert first.status_code == second.status_code == 201
        assert first.headers["cache-control"] == "no-store"
        one, two = first.json(), second.json()
        assert one["secret"].startswith("whsec_") and one["prefix"] == one["secret"][:12]
        assert [s["prefix"] for s in two["active"]] == [two["prefix"], one["prefix"]]
        # listings carry prefixes only: the secret is never shown again
        assert [s["prefix"] for s in shown["webhook_secrets"]] == [two["prefix"], one["prefix"]]
        assert one["secret"] not in str(shown) and two["secret"] not in str(shown)
        assert "sealed" not in str(shown)
        assert retired.status_code == 200
        assert [s["prefix"] for s in retired.json()] == [two["prefix"]]

    async def test_admin_only_and_unknown_tenants(self, admin: httpx.AsyncClient) -> None:
        tenant = await self._tenant(admin)
        url = f"{API}/tenants/{tenant['id']}/webhook-secret"
        wrong = {"Authorization": "Bearer wrong"}

        assert (await admin.post(url, headers=wrong)).status_code == 401
        assert (await admin.delete(f"{url}/previous", headers=wrong)).status_code == 401
        missing = await admin.post(f"{API}/tenants/ten_{'f' * 32}/webhook-secret")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "tenant_not_found"
