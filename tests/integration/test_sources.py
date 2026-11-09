"""SourceRepository and the sources API against real MongoDB."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
import pytest
from asgi_lifespan import LifespanManager

from rag_engine.api import create_app
from rag_engine.api.deps import get_task_queue
from rag_engine.config import Settings
from rag_engine.domain import ResourceStatus
from rag_engine.infrastructure import AppInfrastructure, MongoConnection
from rag_engine.repositories import SourceNotFoundError, SourceRepository
from tests.conftest import APP_STARTUP_TIMEOUT
from tests.support.fakes import RecordingTaskQueue

API = "/api/v1"
SEED = "https://acme.example/"


@dataclass
class _Tenant:
    id: str
    headers: dict[str, str]


@dataclass
class _World:
    client: httpx.AsyncClient
    sources: SourceRepository
    acme: _Tenant
    globex: _Tenant
    tasks: RecordingTaskQueue

    async def kb(self, tenant: _Tenant, name: str = "docs") -> str:
        response = await self.client.post(
            f"{API}/knowledge-bases", json={"name": name}, headers=tenant.headers
        )
        assert response.status_code == 201, response.text
        return response.json()["id"]

    async def source(self, tenant: _Tenant, kb_id: str, seed: str = SEED) -> str:
        source, _ = await self.sources.get_or_create(tenant.id, kb_id, seed)
        return source.id

    def url(self, kb_id: str, source_id: str | None = None) -> str:
        base = f"{API}/knowledge-bases/{kb_id}/sources"
        return f"{base}/{source_id}" if source_id else base


async def _bootstrap(client: httpx.AsyncClient, admin: dict[str, str], name: str) -> _Tenant:
    tenant = (await client.post(f"{API}/admin/tenants", json={"name": name}, headers=admin)).json()
    issued = await client.post(f"{API}/admin/tenants/{tenant['id']}/keys", json={}, headers=admin)
    return _Tenant(tenant["id"], {"Authorization": f"Bearer {issued.json()['api_key']}"})


@pytest.fixture
async def world(integration_settings: Settings) -> AsyncIterator[_World]:
    app = create_app(infrastructure_factory=AppInfrastructure)  # no vectors needed
    tasks = RecordingTaskQueue()
    app.dependency_overrides[get_task_queue] = lambda: tasks
    admin = {"Authorization": f"Bearer {integration_settings.ADMIN_API_KEY.get_secret_value()}"}
    mongo = MongoConnection(integration_settings)
    await mongo.connect()
    transport = httpx.ASGITransport(app=app)
    async with (
        LifespanManager(app, startup_timeout=APP_STARTUP_TIMEOUT),
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
    ):
        yield _World(
            client,
            SourceRepository(mongo.db),
            await _bootstrap(client, admin, "Acme"),
            await _bootstrap(client, admin, "Globex"),
            tasks,
        )
    await mongo.close()


# ---- #
# Tests
# ---- #


class TestSourceRepository_S21:
    """S21 — Source repo + API: identity = seed URL within a KB, race-safe get_or_create."""

    async def test_get_or_create_is_idempotent(self, world: _World) -> None:
        kb = await world.kb(world.acme)

        first, created = await world.sources.get_or_create(world.acme.id, kb, SEED)
        again, created_again = await world.sources.get_or_create(world.acme.id, kb, SEED)

        assert (created, created_again) == (True, False)
        assert again.id == first.id
        assert first.active_version is None and first.stats.pages == 0

    async def test_concurrent_get_or_create_yields_one_source(self, world: _World) -> None:
        kb = await world.kb(world.acme)

        results = await asyncio.gather(
            *(world.sources.get_or_create(world.acme.id, kb, SEED) for _ in range(10))
        )

        assert len({source.id for source, _ in results}) == 1
        assert [created for _, created in results].count(True) == 1
        assert await world.sources.count_active(world.acme.id, kb) == 1

    async def test_same_seed_in_other_kb_or_tenant_is_a_new_source(self, world: _World) -> None:
        kb1, kb2 = await world.kb(world.acme, "one"), await world.kb(world.acme, "two")
        other_kb = await world.kb(world.globex)

        ids = {
            await world.source(world.acme, kb1),
            await world.source(world.acme, kb2),
            await world.source(world.globex, other_kb),
        }

        assert len(ids) == 3

    async def test_deleting_frees_the_seed_url(self, world: _World) -> None:
        kb = await world.kb(world.acme)
        old = await world.source(world.acme, kb)

        await world.sources.mark_deleting(old, world.acme.id, kb)
        new, created = await world.sources.get_or_create(world.acme.id, kb, SEED)

        assert created and new.id != old
        with pytest.raises(SourceNotFoundError):
            await world.sources.mark_deleting(old, world.acme.id, kb)
        stored = await world.sources.get_or_raise(old)
        assert stored.status is ResourceStatus.DELETING


class TestSourceApi_S21:
    """S21 — Source repo + API: list / get / delete, scoped to tenant and knowledge base."""

    async def test_list_and_get(self, world: _World) -> None:
        kb = await world.kb(world.acme)
        ids = [await world.source(world.acme, kb, f"https://site{i}.example/") for i in range(3)]
        headers = world.acme.headers

        first = (await world.client.get(world.url(kb), params={"limit": 2}, headers=headers)).json()
        rest = (
            await world.client.get(
                world.url(kb), params={"limit": 2, "cursor": first["next_cursor"]}, headers=headers
            )
        ).json()
        one = (await world.client.get(world.url(kb, ids[0]), headers=headers)).json()

        assert [s["id"] for s in first["items"] + rest["items"]] == ids[::-1]
        assert one["seed_url"] == "https://site0.example/"
        assert one["kb_id"] == kb and one["status"] == "active"
        assert one["active_version"] is None and one["stats"]["pages"] == 0
        assert "tenant_id" not in one

    async def test_delete_is_202_then_hidden(self, world: _World) -> None:
        kb = await world.kb(world.acme)
        source = await world.source(world.acme, kb)
        url, headers = world.url(kb, source), world.acme.headers

        deleted = await world.client.delete(url, headers=headers)
        again = await world.client.delete(url, headers=headers)
        fetched = await world.client.get(url, headers=headers)
        listed = (await world.client.get(world.url(kb), headers=headers)).json()

        assert deleted.status_code == 202 and deleted.json()["status"] == "deleting"
        assert again.status_code == fetched.status_code == 404
        assert fetched.json()["error"]["code"] == "source_not_found"
        assert listed["items"] == []
        assert world.tasks.sent == [("delete_source", world.acme.id, source)]  # S81 cascade

    async def test_source_is_only_reachable_through_its_own_kb(self, world: _World) -> None:
        kb1, kb2 = await world.kb(world.acme, "one"), await world.kb(world.acme, "two")
        source = await world.source(world.acme, kb1)

        wrong_kb = await world.client.get(world.url(kb2, source), headers=world.acme.headers)
        delete_via_wrong_kb = await world.client.delete(
            world.url(kb2, source), headers=world.acme.headers
        )

        assert wrong_kb.status_code == delete_via_wrong_kb.status_code == 404
        still_there = await world.client.get(world.url(kb1, source), headers=world.acme.headers)
        assert still_there.status_code == 200

    async def test_other_tenant_gets_404_everywhere(self, world: _World) -> None:
        kb = await world.kb(world.acme)
        source = await world.source(world.acme, kb)
        intruder = world.globex.headers

        responses = [
            await world.client.get(world.url(kb), headers=intruder),
            await world.client.get(world.url(kb, source), headers=intruder),
            await world.client.delete(world.url(kb, source), headers=intruder),
        ]

        assert [r.status_code for r in responses] == [404, 404, 404]
        assert {r.json()["error"]["code"] for r in responses} == {"knowledge_base_not_found"}
        assert world.tasks.sent == []  # nothing enqueued for a refused delete

    async def test_deleted_kb_hides_its_sources(self, world: _World) -> None:
        kb = await world.kb(world.acme)
        source = await world.source(world.acme, kb)
        await world.client.delete(f"{API}/knowledge-bases/{kb}", headers=world.acme.headers)

        listed = await world.client.get(world.url(kb), headers=world.acme.headers)
        fetched = await world.client.get(world.url(kb, source), headers=world.acme.headers)

        assert listed.status_code == fetched.status_code == 404

    @pytest.mark.parametrize("path", ["kb_bad/sources", "{kb}/sources/not-an-id"])
    async def test_malformed_ids_are_422(self, world: _World, path: str) -> None:
        kb = await world.kb(world.acme)

        response = await world.client.get(
            f"{API}/knowledge-bases/{path.format(kb=kb)}", headers=world.acme.headers
        )

        assert response.status_code == 422

    async def test_requires_tenant_key(self, world: _World) -> None:
        kb = await world.kb(world.acme)

        assert (await world.client.get(world.url(kb))).status_code == 401
