"""TenantRepository and ApiKeyRepository against real MongoDB."""

from collections.abc import AsyncIterator
from datetime import timedelta

import pytest

from rag_engine.config import Settings
from rag_engine.domain import ApiKey, Tenant, TenantQuotas, TenantStatus, utc_now
from rag_engine.infrastructure import MongoConnection
from rag_engine.repositories import (
    LAST_USED_RESOLUTION,
    ApiKeyNotFoundError,
    ApiKeyRepository,
    DuplicateDocumentError,
    TenantNameTakenError,
    TenantNotFoundError,
    TenantRepository,
)
from rag_engine.security import ApiKeyHasher, parse_api_key


@pytest.fixture
async def mongo(integration_settings: Settings) -> AsyncIterator[MongoConnection]:
    connection = MongoConnection(integration_settings)
    await connection.connect()
    await connection.ensure_indexes([Tenant, ApiKey])
    yield connection
    await connection.close()


@pytest.fixture
def tenants(mongo: MongoConnection) -> TenantRepository:
    return TenantRepository(mongo.db)


@pytest.fixture
def keys(mongo: MongoConnection) -> ApiKeyRepository:
    return ApiKeyRepository(mongo.db)


@pytest.fixture
def hasher(integration_settings: Settings) -> ApiKeyHasher:
    return ApiKeyHasher(integration_settings)


@pytest.fixture
async def tenant(tenants: TenantRepository) -> Tenant:
    return await tenants.create(Tenant(name="Acme"))


class _CollidingHasher(ApiKeyHasher):
    """Returns the same prefix for the first `collisions` calls."""

    def __init__(self, config: Settings, collisions: int) -> None:
        super().__init__(config)
        self.calls = 0
        self.collisions = collisions
        self._first = None

    def generate(self):
        self.calls += 1
        generated = super().generate()
        if self._first is None:
            self._first = generated
        return self._first if self.calls <= self.collisions + 1 else generated


# ---- #
# Tests
# ---- #


class TestTenantRepository_S16:
    """S16 — Tenant & ApiKey repositories: tenants (case-insensitive names, status, quotas)."""

    async def test_create_and_get_by_name_case_insensitively(
        self, tenants: TenantRepository, tenant: Tenant
    ) -> None:
        found = await tenants.get_by_name("  aCmE ")

        assert found == tenant
        assert await tenants.get_by_name("Other") is None

    async def test_duplicate_name_raises_typed_conflict(
        self, tenants: TenantRepository, tenant: Tenant
    ) -> None:
        with pytest.raises(TenantNameTakenError) as exc:
            await tenants.create(Tenant(name="ACME"))

        assert exc.value.status_code == 409
        assert exc.value.details == {"name": "ACME"}

    async def test_rename_into_taken_name_conflicts(
        self, tenants: TenantRepository, tenant: Tenant
    ) -> None:
        other = await tenants.create(Tenant(name="Globex"))

        with pytest.raises(TenantNameTakenError):
            await tenants.update(other.id, {"name": "acme"})
        renamed = await tenants.update(other.id, {"name": "Globex Corp"})
        assert renamed.name == "Globex Corp"

    async def test_set_status_and_update_quotas(
        self, tenants: TenantRepository, tenant: Tenant
    ) -> None:
        disabled = await tenants.set_status(tenant.id, TenantStatus.DISABLED)
        updated = await tenants.update(tenant.id, {"quotas": TenantQuotas(max_concurrent_jobs=5)})

        assert not disabled.is_active
        assert updated.quotas.max_concurrent_jobs == 5
        assert updated.status is TenantStatus.DISABLED

    async def test_unknown_tenant_raises_typed_not_found(self, tenants: TenantRepository) -> None:
        with pytest.raises(TenantNotFoundError) as exc:
            await tenants.set_status(Tenant(name="ghost").id, TenantStatus.DISABLED)

        assert exc.value.code == "tenant_not_found"

    async def test_duplicate_details_name_fields_not_values(
        self, mongo: MongoConnection, tenant: Tenant
    ) -> None:
        from rag_engine.repositories import BaseRepository

        class _RawTenants(BaseRepository[Tenant]):
            model = Tenant

        with pytest.raises(DuplicateDocumentError) as exc:
            await _RawTenants(mongo.db).insert(Tenant(name="acme"))

        assert exc.value.details == {"collection": "tenants", "fields": ["name"]}
        assert "acme" not in str(exc.value.details).lower().replace("tenants", "")


class TestApiKeyRepository_S16:
    """S16 — Tenant & ApiKey repositories: issue, lookup, list, revoke, last-used, rehash."""

    async def test_issue_stores_only_the_hash(
        self, keys: ApiKeyRepository, hasher: ApiKeyHasher, tenant: Tenant, mongo: MongoConnection
    ) -> None:
        record, generated = await keys.issue(tenant.id, hasher, name="ci")

        raw = await mongo.db["api_keys"].find_one({"_id": record.id})
        assert raw is not None
        assert generated.plaintext.get_secret_value() not in str(raw)
        assert raw["key_hash"] == generated.key_hash
        assert record.name == "ci" and record.key_prefix == generated.prefix

    async def test_issued_key_round_trips_through_lookup_and_verify(
        self, keys: ApiKeyRepository, hasher: ApiKeyHasher, tenant: Tenant
    ) -> None:
        _, generated = await keys.issue(tenant.id, hasher)
        parsed = parse_api_key(generated.plaintext.get_secret_value())
        assert parsed is not None

        found = await keys.get_by_prefix(parsed.prefix)

        assert found is not None and found.tenant_id == tenant.id
        assert hasher.verify(parsed.plaintext, found.key_hash).valid
        assert await keys.get_by_prefix("NoSuchPrefix") is None

    async def test_issue_retries_on_prefix_collision(
        self, keys: ApiKeyRepository, tenant: Tenant, integration_settings: Settings
    ) -> None:
        colliding = _CollidingHasher(integration_settings, collisions=2)
        await keys.issue(tenant.id, colliding)  # call 1 stores the shared prefix

        record, _ = await keys.issue(tenant.id, colliding)  # calls 2-3 collide, 4 succeeds

        assert colliding.calls == 4
        assert await keys.count(scope={"tenant_id": tenant.id}) == 2
        assert record.key_prefix != colliding._first.prefix  # type: ignore

    async def test_issue_gives_up_after_repeated_collisions(
        self, keys: ApiKeyRepository, tenant: Tenant, integration_settings: Settings
    ) -> None:
        colliding = _CollidingHasher(integration_settings, collisions=10)
        await keys.issue(tenant.id, colliding)

        with pytest.raises(DuplicateDocumentError):
            await keys.issue(tenant.id, colliding)

    async def test_list_and_count_exclude_revoked_and_expired(
        self,
        keys: ApiKeyRepository,
        hasher: ApiKeyHasher,
        tenants: TenantRepository,
        tenant: Tenant,
    ) -> None:
        now = utc_now()
        active, _ = await keys.issue(tenant.id, hasher, name="active")
        revoked, _ = await keys.issue(tenant.id, hasher, name="revoked")
        await keys.issue(tenant.id, hasher, name="expiring", expires_at=now + timedelta(hours=1))
        await keys.revoke(revoked.id, tenant.id)
        other = await tenants.create(Tenant(name="Other"))
        await keys.issue(other.id, hasher)

        listed = await keys.list_for_tenant(tenant.id)
        everything = await keys.list_for_tenant(tenant.id, include_revoked=True)

        assert {k.name for k in listed.items} == {"active", "expiring"}
        assert len(everything.items) == 3
        assert await keys.count_active(tenant.id, now) == 2
        assert await keys.count_active(tenant.id, now + timedelta(hours=2)) == 1
        assert active.id in {k.id for k in listed.items}

    async def test_revoke_is_idempotent_and_tenant_scoped(
        self,
        keys: ApiKeyRepository,
        hasher: ApiKeyHasher,
        tenants: TenantRepository,
        tenant: Tenant,
    ) -> None:
        record, _ = await keys.issue(tenant.id, hasher)
        other = await tenants.create(Tenant(name="Other"))

        with pytest.raises(ApiKeyNotFoundError):
            await keys.revoke(record.id, other.id)
        first = await keys.revoke(record.id, tenant.id)
        second = await keys.revoke(record.id, tenant.id)

        assert first.revoked_at is not None
        assert second.revoked_at == first.revoked_at
        assert not second.is_usable()

    async def test_touch_last_used_is_throttled(
        self, keys: ApiKeyRepository, hasher: ApiKeyHasher, tenant: Tenant
    ) -> None:
        record, _ = await keys.issue(tenant.id, hasher)
        t0 = utc_now()

        wrote = [
            await keys.touch_last_used(record.id, t0),
            await keys.touch_last_used(record.id, t0 + timedelta(seconds=10)),
            await keys.touch_last_used(record.id, t0 + LAST_USED_RESOLUTION + timedelta(seconds=1)),
        ]

        assert wrote == [True, False, True]
        stored = await keys.get_or_raise(record.id)
        assert stored.last_used_at == t0 + LAST_USED_RESOLUTION + timedelta(seconds=1)

    async def test_rehash_after_pepper_rotation(
        self, keys: ApiKeyRepository, tenant: Tenant, override_settings
    ) -> None:
        old_pepper = "old-pepper-0123456789abcdef0123456789abcdef"
        old_hasher = ApiKeyHasher(override_settings(API_KEY_PEPPER=old_pepper))
        record, generated = await keys.issue(tenant.id, old_hasher)
        rotated = ApiKeyHasher(
            override_settings(
                API_KEY_PEPPER="new-pepper-0123456789abcdef0123456789abcdef",
                API_KEY_PEPPER_PREVIOUS=[old_pepper],
            )
        )
        assert rotated.verify(generated.plaintext, record.key_hash).needs_rehash

        await keys.rehash(record.id, generated.plaintext, rotated)

        stored = await keys.get_or_raise(record.id)
        result = rotated.verify(generated.plaintext, stored.key_hash)
        assert result.valid and not result.needs_rehash
