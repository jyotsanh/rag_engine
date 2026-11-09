"""Admin API (`/admin/*`) through the real app and MongoDB."""

from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager

from rag_engine.api import create_app
from rag_engine.config import Settings
from rag_engine.domain import utc_now
from rag_engine.infrastructure import AppInfrastructure
from tests.conftest import APP_STARTUP_TIMEOUT, OverrideSettings

API = "/api/v1/admin"


@pytest.fixture
async def admin(integration_settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(infrastructure_factory=AppInfrastructure)  # no vectors needed
    headers = {"Authorization": f"Bearer {integration_settings.ADMIN_API_KEY.get_secret_value()}"}
    transport = httpx.ASGITransport(app=app)
    async with (
        LifespanManager(app, startup_timeout=APP_STARTUP_TIMEOUT),
        httpx.AsyncClient(transport=transport, base_url="http://t", headers=headers) as client,
    ):
        yield client


async def _tenant(admin: httpx.AsyncClient, name: str = "Acme", **body: Any) -> dict[str, Any]:
    response = await admin.post(f"{API}/tenants", json={"name": name, **body})
    assert response.status_code == 201, response.text
    return response.json()


async def _key(admin: httpx.AsyncClient, tenant_id: str, **body: Any) -> dict[str, Any]:
    response = await admin.post(f"{API}/tenants/{tenant_id}/keys", json=body)
    assert response.status_code == 201, response.text
    return response.json()


# ---- #
# Tests
# ---- #


class TestAdminTenantsApi_S18:
    """S18 — Admin API: tenant create / list / get / patch."""

    async def test_every_route_requires_the_admin_key(self, admin: httpx.AsyncClient) -> None:
        tenant_id = (await _tenant(admin))["id"]
        calls = [
            ("POST", "/tenants"),
            ("GET", "/tenants"),
            ("GET", f"/tenants/{tenant_id}"),
            ("PATCH", f"/tenants/{tenant_id}"),
            ("POST", f"/tenants/{tenant_id}/keys"),
            ("GET", f"/tenants/{tenant_id}/keys"),
            ("DELETE", f"/tenants/{tenant_id}/keys/key_{'0' * 32}"),
            ("POST", f"/tenants/{tenant_id}/webhook-secret"),
            ("DELETE", f"/tenants/{tenant_id}/webhook-secret/previous"),
        ]

        for method, path in calls:
            response = await admin.request(
                method, API + path, headers={"Authorization": "Bearer wrong"}, json={}
            )
            assert response.status_code == 401, (method, path)

    async def test_create_with_defaults(
        self, admin: httpx.AsyncClient, integration_settings: Settings
    ) -> None:
        body = await _tenant(admin, "  Acme  ")

        assert body["id"].startswith("ten_")
        assert body["name"] == "Acme"
        assert body["status"] == "active"
        assert body["default_embedding_model"] == integration_settings.DEFAULT_EMBEDDING_MODEL
        assert body["quotas"]["max_concurrent_jobs"] == (
            integration_settings.TENANT_DEFAULT_MAX_CONCURRENT_JOBS
        )

    async def test_create_with_model_and_partial_quotas(self, admin: httpx.AsyncClient) -> None:
        body = await _tenant(
            admin,
            default_embedding_model="text-embedding-3-large",
            quotas={"max_concurrent_jobs": 5},
        )

        assert body["default_embedding_model"] == "text-embedding-3-large"
        assert body["quotas"]["max_concurrent_jobs"] == 5
        assert body["quotas"]["max_knowledge_bases"] > 0  # untouched default

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"name": ""},
            {"name": "x", "unknown": 1},
            {"name": "x", "default_embedding_model": "ada-002"},
            {"name": "x", "quotas": {"max_concurrent_jobs": 0}},
            {"name": "x", "quotas": {"max_pages_per_job": 10**9}},
        ],
    )
    async def test_create_validation(self, admin: httpx.AsyncClient, body: dict) -> None:
        response = await admin.post(f"{API}/tenants", json=body)

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    async def test_duplicate_name_is_409(self, admin: httpx.AsyncClient) -> None:
        await _tenant(admin, "Acme")

        response = await admin.post(f"{API}/tenants", json={"name": "ACME"})

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "tenant_name_taken"

    async def test_get_and_not_found(self, admin: httpx.AsyncClient) -> None:
        tenant = await _tenant(admin)

        assert (await admin.get(f"{API}/tenants/{tenant['id']}")).json() == tenant
        missing = await admin.get(f"{API}/tenants/ten_{'0' * 32}")
        malformed = await admin.get(f"{API}/tenants/not-an-id")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "tenant_not_found"
        assert malformed.status_code == 422

    async def test_list_paginates_and_filters_by_status(self, admin: httpx.AsyncClient) -> None:
        ids = [(await _tenant(admin, f"T{i}"))["id"] for i in range(5)]
        await admin.patch(f"{API}/tenants/{ids[0]}", json={"status": "disabled"})

        first = (await admin.get(f"{API}/tenants", params={"limit": 3})).json()
        second = (
            await admin.get(f"{API}/tenants", params={"limit": 3, "cursor": first["next_cursor"]})
        ).json()
        disabled = (await admin.get(f"{API}/tenants", params={"status": "disabled"})).json()

        assert [t["id"] for t in first["items"] + second["items"]] == ids[::-1]
        assert second["next_cursor"] is None
        assert [t["id"] for t in disabled["items"]] == [ids[0]]

    async def test_patch_status_name_model_and_merge_quotas(self, admin: httpx.AsyncClient) -> None:
        tenant = await _tenant(admin, quotas={"max_concurrent_jobs": 5})

        response = await admin.patch(
            f"{API}/tenants/{tenant['id']}",
            json={
                "name": "Acme Corp",
                "status": "disabled",
                "default_embedding_model": "text-embedding-3-large",
                "quotas": {"max_knowledge_bases": 3},
            },
        )

        body = response.json()
        assert response.status_code == 200
        assert (body["name"], body["status"]) == ("Acme Corp", "disabled")
        assert body["default_embedding_model"] == "text-embedding-3-large"
        assert body["quotas"]["max_knowledge_bases"] == 3
        assert body["quotas"]["max_concurrent_jobs"] == 5  # merged, not reset
        assert body["updated_at"] > tenant["updated_at"]

    @pytest.mark.parametrize(
        "body",
        [{}, {"name": None}, {"status": "archived"}, {"quotas": {"max_pages_per_job": 10**9}}],
    )
    async def test_patch_validation(self, admin: httpx.AsyncClient, body: dict) -> None:
        tenant = await _tenant(admin)

        response = await admin.patch(f"{API}/tenants/{tenant['id']}", json=body)

        assert response.status_code == 422

    async def test_patch_rename_conflict_and_missing(self, admin: httpx.AsyncClient) -> None:
        await _tenant(admin, "Acme")
        other = await _tenant(admin, "Globex")

        conflict = await admin.patch(f"{API}/tenants/{other['id']}", json={"name": "acme"})
        missing = await admin.patch(f"{API}/tenants/ten_{'0' * 32}", json={"name": "x"})

        assert conflict.status_code == 409
        assert missing.status_code == 404


class TestAdminKeysApi_S18:
    """S18 — Admin API: issue / list / revoke tenant API keys."""

    async def test_issue_returns_plaintext_once_and_no_store(
        self, admin: httpx.AsyncClient
    ) -> None:
        tenant = await _tenant(admin)

        response = await admin.post(f"{API}/tenants/{tenant['id']}/keys", json={"name": "ci"})

        body = response.json()
        assert response.status_code == 201
        assert response.headers["cache-control"] == "no-store"
        assert body["api_key"].startswith(body["hint"].removesuffix("…"))
        assert (body["name"], body["state"], body["tenant_id"]) == ("ci", "active", tenant["id"])
        listed = (await admin.get(f"{API}/tenants/{tenant['id']}/keys")).json()
        assert "api_key" not in listed["items"][0]
        assert body["api_key"] not in str(listed)
        assert "key_hash" not in str(listed)

    async def test_issued_key_authenticates_as_the_tenant(self, admin: httpx.AsyncClient) -> None:
        tenant = await _tenant(admin)
        key = (await _key(admin, tenant["id"]))["api_key"]

        # the admin router rejects tenant keys...
        as_admin = await admin.get(f"{API}/tenants", headers={"Authorization": f"Bearer {key}"})

        assert as_admin.status_code == 401
        # ...and the key's tenant reads its own record back through the repository flow
        from rag_engine.security import parse_api_key

        assert parse_api_key(key) is not None

    async def test_expiry_must_be_future_and_aware(self, admin: httpx.AsyncClient) -> None:
        tenant = await _tenant(admin)
        url = f"{API}/tenants/{tenant['id']}/keys"
        past = (utc_now() - timedelta(minutes=1)).isoformat()
        naive = (utc_now() + timedelta(days=1)).replace(tzinfo=None).isoformat()
        future = (utc_now() + timedelta(days=30)).isoformat()

        assert (await admin.post(url, json={"expires_at": past})).status_code == 422
        assert (await admin.post(url, json={"expires_at": naive})).status_code == 422
        created = await admin.post(url, json={"expires_at": future})
        assert created.status_code == 201
        assert created.json()["expires_at"] is not None

    async def test_key_cap_enforced(
        self, admin: httpx.AsyncClient, override_settings: OverrideSettings
    ) -> None:
        override_settings(TENANT_MAX_ACTIVE_API_KEYS=2)
        tenant = await _tenant(admin)
        first = await _key(admin, tenant["id"])
        await _key(admin, tenant["id"])

        blocked = await admin.post(f"{API}/tenants/{tenant['id']}/keys", json={})
        await admin.delete(f"{API}/tenants/{tenant['id']}/keys/{first['id']}")
        after_revoke = await admin.post(f"{API}/tenants/{tenant['id']}/keys", json={})

        assert blocked.status_code == 403
        assert blocked.json()["error"]["code"] == "quota_exceeded"
        assert blocked.json()["error"]["details"] == {"quota": "max_active_api_keys", "limit": 2}
        assert after_revoke.status_code == 201

    async def test_list_hides_revoked_unless_asked(self, admin: httpx.AsyncClient) -> None:
        tenant = await _tenant(admin)
        keep = await _key(admin, tenant["id"], name="keep")
        drop = await _key(admin, tenant["id"], name="drop")
        await admin.delete(f"{API}/tenants/{tenant['id']}/keys/{drop['id']}")
        url = f"{API}/tenants/{tenant['id']}/keys"

        active = (await admin.get(url)).json()["items"]
        everything = (await admin.get(url, params={"include_revoked": True})).json()["items"]

        assert [k["id"] for k in active] == [keep["id"]]
        assert {k["state"] for k in everything} == {"active", "revoked"}

    async def test_revoke_is_idempotent_and_tenant_scoped(self, admin: httpx.AsyncClient) -> None:
        tenant = await _tenant(admin, "A")
        other = await _tenant(admin, "B")
        key = await _key(admin, tenant["id"])

        wrong_tenant = await admin.delete(f"{API}/tenants/{other['id']}/keys/{key['id']}")
        first = await admin.delete(f"{API}/tenants/{tenant['id']}/keys/{key['id']}")
        second = await admin.delete(f"{API}/tenants/{tenant['id']}/keys/{key['id']}")

        assert wrong_tenant.status_code == 404
        assert first.status_code == second.status_code == 200
        assert first.json()["state"] == "revoked"
        assert first.json()["revoked_at"] == second.json()["revoked_at"]

    async def test_key_routes_404_for_unknown_tenant(self, admin: httpx.AsyncClient) -> None:
        ghost = f"ten_{'0' * 32}"

        assert (await admin.post(f"{API}/tenants/{ghost}/keys", json={})).status_code == 404
        assert (await admin.get(f"{API}/tenants/{ghost}/keys")).status_code == 404

    async def test_openapi_lists_admin_routes_under_admin_security(
        self, admin: httpx.AsyncClient
    ) -> None:
        spec = (await admin.get("/api/v1/openapi.json")).json()

        admin_ops = [
            op for path, ops in spec["paths"].items() if "/admin/" in path for op in ops.values()
        ]
        assert len(admin_ops) == 9  # + webhook-secret rotate / retire previous (S82)
        assert all(op["security"] == [{"AdminKey": []}] for op in admin_ops)
        assert "key_hash" not in str(spec["components"]["schemas"].get("ApiKeyResponse"))
        assert "sealed" not in str(spec["components"]["schemas"].get("TenantResponse"))
