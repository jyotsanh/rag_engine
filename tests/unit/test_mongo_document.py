"""MongoDocument base + pagination cursors + lifespan wiring (no database needed)."""

import uuid
from datetime import UTC, datetime

import pytest
from asgi_lifespan import LifespanManager
from pydantic import ValidationError
from pymongo import ASCENDING, IndexModel

from rag_engine.api import create_app
from rag_engine.domain import MongoDocument, registered_documents, utc_now
from rag_engine.repositories import InvalidCursorError
from rag_engine.repositories.pagination import decode_cursor, encode_cursor
from tests.support.fakes import NoopInfrastructure


class _Gadget(MongoDocument):
    collection_name = "s10_unit_gadgets"
    id_prefix = "gad"
    indexes = (IndexModel([("sku", ASCENDING)], unique=True),)

    sku: str
    quantity: int = 0


class _AbstractBase(MongoDocument):
    """No collection_name: an intermediate class that must not register."""

    owner: str = "x"


class _FailingInfrastructure(NoopInfrastructure):
    async def start(self) -> None:
        raise RuntimeError("mongo down")


# ---- #
# Tests
# ---- #


class TestMongoDocument_S10:
    """S10 — Mongo infrastructure: MongoDocument ids, BSON mapping, registry, cursors."""

    def test_id_is_prefixed_uuid7(self) -> None:
        gadget = _Gadget(sku="A")

        prefix, hex_ = gadget.id.split("_", 1)
        assert prefix == "gad"
        assert uuid.UUID(hex_).version == 7

    def test_ids_sort_by_creation_order(self) -> None:
        ids = [_Gadget(sku=str(i)).id for i in range(50)]

        assert ids == sorted(ids)

    def test_explicit_id_is_kept(self) -> None:
        assert _Gadget(sku="A", id="gad_fixed").id == "gad_fixed"

    def test_to_mongo_uses_underscore_id_and_native_datetimes(self) -> None:
        gadget = _Gadget(sku="A", quantity=2)

        raw = gadget.to_mongo()

        assert raw["_id"] == gadget.id
        assert "id" not in raw
        assert isinstance(raw["created_at"], datetime)
        assert raw["created_at"].tzinfo is UTC

    def test_from_mongo_round_trips_and_ignores_unknown_fields(self) -> None:
        gadget = _Gadget(sku="A", quantity=2)

        loaded = _Gadget.from_mongo({**gadget.to_mongo(), "added_by_newer_code": 1})

        assert loaded == gadget

    def test_timestamps_are_utc_millisecond_precision(self) -> None:
        now = utc_now()

        assert now.tzinfo is UTC
        assert now.microsecond % 1000 == 0

    def test_assignment_is_validated(self) -> None:
        gadget = _Gadget(sku="A")

        with pytest.raises(ValidationError):
            gadget.quantity = "lots"  # type: ignore

    def test_concrete_subclasses_register_but_abstract_ones_do_not(self) -> None:
        registered = registered_documents()

        assert _Gadget in registered
        assert _AbstractBase not in registered

    def test_two_models_cannot_share_a_collection(self) -> None:
        with pytest.raises(TypeError, match="s10_unit_gadgets"):

            class _Clash(MongoDocument):
                collection_name = "s10_unit_gadgets"
                id_prefix = "cl"

    @pytest.mark.parametrize("order", ["asc", "desc"])
    def test_cursor_round_trip(self, order: str) -> None:
        cursor = encode_cursor("gad_123", order)  # type: ignore

        assert decode_cursor(cursor, order) == "gad_123"  # type: ignore
        assert "=" not in cursor

    @pytest.mark.parametrize("cursor", ["not-base64!!", "e30", "eyJhZnRlciI6IDF9", ""])
    def test_malformed_cursor_rejected(self, cursor: str) -> None:
        with pytest.raises(InvalidCursorError):
            decode_cursor(cursor, "desc")

    def test_cursor_from_other_order_rejected(self) -> None:
        with pytest.raises(InvalidCursorError):
            decode_cursor(encode_cursor("gad_1", "asc"), "desc")

    async def test_lifespan_starts_and_stops_infrastructure(self) -> None:
        infra = NoopInfrastructure()
        app = create_app(lambda: infra)

        async with LifespanManager(app):
            assert infra.started
            assert app.state.infrastructure is infra
            assert not infra.stopped

        assert infra.stopped

    async def test_lifespan_aborts_when_infrastructure_fails(self) -> None:
        app = create_app(_FailingInfrastructure)

        with pytest.raises(RuntimeError, match="mongo down"):
            async with LifespanManager(app):
                pass
