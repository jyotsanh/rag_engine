"""Knowledge base API through the real app, as two tenants bootstrapped via the admin API."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager

from rag_engine.api import create_app
from rag_engine.config import Settings

API = "/api/v1"
KBS = f"{API}/knowledge-bases"


@dataclass
class _Tenant:
    id: str
    headers: dict[str, str]


@dataclass
class _World:
    client: httpx.AsyncClient
    admin: dict[str, str]
    acme: _Tenant
    globex: _Tenant

    async def create_kb(self, tenant: _Tenant, **body: Any) -> dict[str, Any]:
        response = await self.client.post(
            KBS, json={"name": "docs", **body}, headers=tenant.headers
        )
        assert response.status_code == 201, response.text
        return response.json()


async def _bootstrap_tenant(
    client: httpx.AsyncClient, admin: dict[str, str], name: str, **body: Any
) -> _Tenant:
    tenant = (
        await client.post(f"{API}/admin/tenants", json={"name": name, **body}, headers=admin)
    ).json()
    key = (
        await client.post(f"{API}/admin/tenants/{tenant['id']}/keys", json={}, headers=admin)
    ).json()
    return _Tenant(tenant["id"], {"Authorization": f"Bearer {key['api_key']}"})


@pytest.fixture
async def world(integration_settings: Settings) -> AsyncIterator[_World]:
    app = create_app()
    admin = {"Authorization": f"Bearer {integration_settings.ADMIN_API_KEY.get_secret_value()}"}
    transport = httpx.ASGITransport(app=app)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
    ):
        acme = await _bootstrap_tenant(client, admin, "Acme", quotas={"max_knowledge_bases": 3})
        globex = await _bootstrap_tenant(client, admin, "Globex")
        yield _World(client, admin, acme, globex)


# ---- #
# Tests
# ---- #


class TestKnowledgeBaseApi_S20:
    """S20 — KB API: create / list / get / patch / delete and tenant isolation."""

    async def test_requires_a_tenant_key(self, world: _World) -> None:
        no_key = await world.client.get(KBS)
        admin_key = await world.client.get(KBS, headers=world.admin)

        assert no_key.status_code == admin_key.status_code == 401

    async def test_create_returns_locked_embedding_space(self, world: _World) -> None:
        body = await world.create_kb(
            world.acme,
            name="handbook",
            description="HR policies",
            embedding_model="text-embedding-3-large",
            embedding_dimensions=1024,
        )

        assert body["id"].startswith("kb_")
        assert (body["name"], body["description"], body["status"]) == (
            "handbook",
            "HR policies",
            "active",
        )
        assert (body["embedding_model"], body["embedding_dimensions"]) == (
            "text-embedding-3-large",
            1024,
        )
        assert "tenant_id" not in body

    async def test_create_defaults_to_tenant_model_native_size(
        self, world: _World, integration_settings: Settings
    ) -> None:
        body = await world.create_kb(world.acme)

        assert body["embedding_model"] == integration_settings.DEFAULT_EMBEDDING_MODEL
        assert body["embedding_dimensions"] == 1536

    @pytest.mark.parametrize(
        ("body", "code"),
        [
            ({"name": ""}, "validation_error"),
            ({"name": "x", "tenant_id": "ten_x"}, "validation_error"),
            ({"name": "x", "embedding_model": "ada-002"}, "validation_error"),
            ({"name": "x", "embedding_dimensions": 0}, "validation_error"),
            ({"name": "x", "embedding_dimensions": 100}, "unsupported_embedding_dimensions"),
            ({"name": "x", "embedding_dimensions": 4000}, "unsupported_embedding_dimensions"),
        ],
    )
    async def test_create_validation(self, world: _World, body: dict, code: str) -> None:
        response = await world.client.post(KBS, json=body, headers=world.acme.headers)

        assert response.status_code == 422
        assert response.json()["error"]["code"] == code

    async def test_duplicate_name_and_quota(self, world: _World) -> None:
        await world.create_kb(world.acme, name="a")

        duplicate = await world.client.post(KBS, json={"name": "a"}, headers=world.acme.headers)
        await world.create_kb(world.acme, name="b")
        await world.create_kb(world.acme, name="c")
        over_quota = await world.client.post(KBS, json={"name": "d"}, headers=world.acme.headers)

        assert duplicate.status_code == 409
        assert duplicate.json()["error"]["code"] == "knowledge_base_name_taken"
        assert over_quota.status_code == 403
        assert over_quota.json()["error"]["details"] == {
            "quota": "max_knowledge_bases",
            "limit": 3,
        }

    async def test_list_get_and_pagination(self, world: _World) -> None:
        created = [await world.create_kb(world.acme, name=f"kb{i}") for i in range(3)]

        first = (
            await world.client.get(KBS, params={"limit": 2}, headers=world.acme.headers)
        ).json()
        rest = (
            await world.client.get(
                KBS, params={"limit": 2, "cursor": first["next_cursor"]}, headers=world.acme.headers
            )
        ).json()
        one = await world.client.get(f"{KBS}/{created[0]['id']}", headers=world.acme.headers)

        assert [k["id"] for k in first["items"] + rest["items"]] == [c["id"] for c in created][::-1]
        assert rest["next_cursor"] is None
        assert one.json() == created[0]

    async def test_patch_name_and_description(self, world: _World) -> None:
        kb = await world.create_kb(world.acme)
        url = f"{KBS}/{kb['id']}"

        renamed = await world.client.patch(
            url, json={"name": "manuals"}, headers=world.acme.headers
        )
        locked = await world.client.patch(
            url, json={"embedding_model": "text-embedding-3-large"}, headers=world.acme.headers
        )
        empty = await world.client.patch(url, json={}, headers=world.acme.headers)
        null = await world.client.patch(url, json={"name": None}, headers=world.acme.headers)

        assert renamed.status_code == 200 and renamed.json()["name"] == "manuals"
        assert renamed.json()["embedding_model"] == kb["embedding_model"]
        assert locked.status_code == empty.status_code == null.status_code == 422

    async def test_delete_is_202_hides_kb_and_frees_name(self, world: _World) -> None:
        kb = await world.create_kb(world.acme)
        url = f"{KBS}/{kb['id']}"

        deleted = await world.client.delete(url, headers=world.acme.headers)
        again = await world.client.delete(url, headers=world.acme.headers)
        fetched = await world.client.get(url, headers=world.acme.headers)
        listed = (await world.client.get(KBS, headers=world.acme.headers)).json()
        recreated = await world.client.post(KBS, json={"name": "docs"}, headers=world.acme.headers)

        assert deleted.status_code == 202 and deleted.json()["status"] == "deleting"
        assert again.status_code == fetched.status_code == 404
        assert listed["items"] == []
        assert recreated.status_code == 201

    async def test_malformed_ids_are_422(self, world: _World) -> None:
        for url in (f"{KBS}/not-an-id", f"{KBS}/ten_{'0' * 32}"):
            assert (await world.client.get(url, headers=world.acme.headers)).status_code == 422


class TestKnowledgeBaseTenantIsolation_S20:
    """S20 — KB API: one tenant can never see or touch another tenant's knowledge bases."""

    async def test_other_tenant_gets_404_on_every_route(self, world: _World) -> None:
        kb = await world.create_kb(world.acme)
        url = f"{KBS}/{kb['id']}"
        intruder = world.globex.headers

        responses = [
            await world.client.get(url, headers=intruder),
            await world.client.patch(url, json={"name": "pwned"}, headers=intruder),
            await world.client.delete(url, headers=intruder),
        ]

        assert [r.status_code for r in responses] == [404, 404, 404]
        # indistinguishable from a KB that never existed
        missing = await world.client.get(f"{KBS}/kb_{'0' * 32}", headers=intruder)
        assert responses[0].json()["error"]["code"] == missing.json()["error"]["code"]
        owner_view = (await world.client.get(url, headers=world.acme.headers)).json()
        assert owner_view == kb

    async def test_listings_are_tenant_scoped(self, world: _World) -> None:
        mine = await world.create_kb(world.acme, name="shared-name")
        theirs = await world.create_kb(world.globex, name="shared-name")

        acme_list = (await world.client.get(KBS, headers=world.acme.headers)).json()["items"]
        globex_list = (await world.client.get(KBS, headers=world.globex.headers)).json()["items"]

        assert [k["id"] for k in acme_list] == [mine["id"]]
        assert [k["id"] for k in globex_list] == [theirs["id"]]

    async def test_quota_is_per_tenant(self, world: _World) -> None:
        for i in range(3):
            await world.create_kb(world.acme, name=f"a{i}")

        response = await world.client.post(KBS, json={"name": "g"}, headers=world.globex.headers)

        assert response.status_code == 201

    async def test_disabled_tenant_is_403(self, world: _World) -> None:
        await world.client.patch(
            f"{API}/admin/tenants/{world.acme.id}", json={"status": "disabled"}, headers=world.admin
        )

        response = await world.client.get(KBS, headers=world.acme.headers)

        assert response.status_code == 403
        assert response.json()["error"]["code"] == "tenant_disabled"
