"""MongoConnection + BaseRepository against a real MongoDB container."""

from collections.abc import AsyncIterator
from typing import ClassVar

import pytest
from asgi_lifespan import LifespanManager
from pymongo import ASCENDING, IndexModel

from rag_engine.api import create_app
from rag_engine.config import Settings
from rag_engine.domain import MongoDocument
from rag_engine.exceptions import AppValidationError
from rag_engine.infrastructure import AppInfrastructure, DatabaseUnavailableError, MongoConnection
from rag_engine.repositories import (
    BaseRepository,
    DocumentNotFoundError,
    DuplicateDocumentError,
    InvalidCursorError,
)
from tests.conftest import OverrideSettings


class _Widget(MongoDocument):
    collection_name = "s10_widgets"
    id_prefix = "wid"
    indexes = (
        IndexModel([("tenant_id", ASCENDING), ("sku", ASCENDING)], unique=True, name="tenant_sku"),
    )

    tenant_id: str
    sku: str
    quantity: int = 0


class _WidgetNotFound(DocumentNotFoundError):
    code = "widget_not_found"


class _WidgetRepository(BaseRepository[_Widget]):
    model: ClassVar[type[MongoDocument]] = _Widget
    not_found_error = _WidgetNotFound


@pytest.fixture
async def mongo(integration_settings: Settings) -> AsyncIterator[MongoConnection]:
    connection = MongoConnection(integration_settings)
    await connection.connect()
    await connection.ensure_indexes([_Widget])
    yield connection
    await connection.close()


@pytest.fixture
def repo(mongo: MongoConnection) -> _WidgetRepository:
    return _WidgetRepository(mongo.db)


# ---- #
# Tests
# ---- #


class TestMongoInfrastructure_S10:
    """S10 — Mongo infrastructure: connection lifecycle, indexes, BaseRepository, pagination."""

    async def test_connect_ping_close(self, integration_settings: Settings) -> None:
        connection = MongoConnection(integration_settings)

        await connection.connect()
        await connection.ping()
        await connection.close()

        with pytest.raises(DatabaseUnavailableError):
            _ = connection.client

    async def test_unreachable_server_fails_fast(self, override_settings: OverrideSettings) -> None:
        config = override_settings(
            MONGO_URI="mongodb://127.0.0.1:1", MONGO_SERVER_SELECTION_TIMEOUT_MS=300
        )
        connection = MongoConnection(config)

        with pytest.raises(DatabaseUnavailableError):
            await connection.connect()
        with pytest.raises(DatabaseUnavailableError):
            _ = connection.client  # failed connect leaves no half-open client

    async def test_ensure_indexes_is_idempotent(self, mongo: MongoConnection) -> None:
        first = await mongo.ensure_indexes([_Widget])
        second = await mongo.ensure_indexes([_Widget])

        assert first == second == {"s10_widgets": ["tenant_sku"]}
        info = await mongo.db["s10_widgets"].index_information()
        assert info["tenant_sku"]["unique"] is True

    async def test_insert_and_get_round_trip(self, repo: _WidgetRepository) -> None:
        widget = await repo.insert(_Widget(tenant_id="t1", sku="A", quantity=3))

        loaded = await repo.get(widget.id)

        assert loaded == widget

    async def test_unique_index_raises_duplicate_error(self, repo: _WidgetRepository) -> None:
        await repo.insert(_Widget(tenant_id="t1", sku="A"))

        with pytest.raises(DuplicateDocumentError) as exc:
            await repo.insert(_Widget(tenant_id="t1", sku="A"))

        assert exc.value.status_code == 409
        await repo.insert(_Widget(tenant_id="t2", sku="A"))  # other tenant is fine

    async def test_get_or_raise_uses_repository_error(self, repo: _WidgetRepository) -> None:
        with pytest.raises(_WidgetNotFound) as exc:
            await repo.get_or_raise("wid_missing")

        assert exc.value.details == {"id": "wid_missing"}
        assert exc.value.code == "widget_not_found"

    async def test_scope_isolates_tenants(self, repo: _WidgetRepository) -> None:
        mine = await repo.insert(_Widget(tenant_id="t1", sku="A"))
        await repo.insert(_Widget(tenant_id="t2", sku="B"))
        other_scope = {"tenant_id": "t2"}

        assert await repo.get(mine.id, scope=other_scope) is None
        assert await repo.count(scope={"tenant_id": "t1"}) == 1
        with pytest.raises(_WidgetNotFound):
            await repo.update(mine.id, {"quantity": 9}, scope=other_scope)
        assert await repo.delete(mine.id, scope=other_scope) is False
        assert await repo.get(mine.id) is not None

    async def test_update_sets_fields_and_bumps_updated_at(self, repo: _WidgetRepository) -> None:
        widget = await repo.insert(_Widget(tenant_id="t1", sku="A"))

        updated = await repo.update(widget.id, {"quantity": 7})

        assert updated.quantity == 7
        assert updated.created_at == widget.created_at
        assert updated.updated_at >= widget.updated_at
        assert await repo.get(widget.id) == updated

    async def test_update_validates_values(self, repo: _WidgetRepository) -> None:
        widget = await repo.insert(_Widget(tenant_id="t1", sku="A"))

        with pytest.raises(AppValidationError) as invalid:
            await repo.update(widget.id, {"quantity": "many"})
        assert invalid.value.details["errors"][0]["loc"] == ("quantity",)
        with pytest.raises(ValueError, match="immutable"):
            await repo.update(widget.id, {"created_at": None})

        assert (await repo.get_or_raise(widget.id)).quantity == 0

    async def test_update_into_unique_conflict(self, repo: _WidgetRepository) -> None:
        await repo.insert(_Widget(tenant_id="t1", sku="A"))
        b = await repo.insert(_Widget(tenant_id="t1", sku="B"))

        with pytest.raises(DuplicateDocumentError):
            await repo.update(b.id, {"sku": "A"})

    async def test_find_sort_limit_and_delete_many(self, repo: _WidgetRepository) -> None:
        for i in range(5):
            await repo.insert(_Widget(tenant_id="t1", sku=f"S{i}", quantity=i))

        top = await repo.find({"quantity": {"$gte": 2}}, sort=[("quantity", -1)], limit=2)
        deleted = await repo.delete_many({"quantity": {"$lt": 2}})

        assert [w.quantity for w in top] == [4, 3]
        assert deleted == 2
        assert await repo.count() == 3
        with pytest.raises(ValueError, match="refusing"):
            await repo.delete_many({})

    @pytest.mark.parametrize("order", ["desc", "asc"])
    async def test_paginate_walks_every_document_once(
        self, repo: _WidgetRepository, order: str
    ) -> None:
        inserted = [await repo.insert(_Widget(tenant_id="t1", sku=f"S{i}")) for i in range(7)]
        await repo.insert(_Widget(tenant_id="t2", sku="other"))

        seen: list[str] = []
        pages = 0
        cursor = None
        while True:
            page = await repo.paginate(
                scope={"tenant_id": "t1"},
                limit=3,
                cursor=cursor,
                order=order,  # type: ignore
            )
            seen += [w.id for w in page.items]
            pages += 1
            if page.next_cursor is None:
                break
            cursor = page.next_cursor

        expected = [w.id for w in inserted]
        assert seen == (expected[::-1] if order == "desc" else expected)
        assert pages == 3

    async def test_paginate_exact_multiple_has_no_empty_trailing_page(
        self, repo: _WidgetRepository
    ) -> None:
        for i in range(4):
            await repo.insert(_Widget(tenant_id="t1", sku=f"S{i}"))

        first = await repo.paginate(limit=2)
        assert first.next_cursor is not None
        second = await repo.paginate(limit=2, cursor=first.next_cursor)

        assert len(second.items) == 2
        assert second.next_cursor is None

    async def test_paginate_rejects_bad_input(self, repo: _WidgetRepository) -> None:
        with pytest.raises(ValueError, match="limit"):
            await repo.paginate(limit=0)
        with pytest.raises(InvalidCursorError):
            await repo.paginate(cursor="garbage")

    async def test_app_lifespan_connects_and_creates_indexes(
        self, integration_settings: Settings
    ) -> None:
        app = create_app()

        async with LifespanManager(app):
            infra = app.state.infrastructure
            assert isinstance(infra, AppInfrastructure)
            info = await infra.mongo.db["s10_widgets"].index_information()
            assert "tenant_sku" in info

        with pytest.raises(DatabaseUnavailableError):
            _ = infra.mongo.client
