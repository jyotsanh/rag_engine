"""Tenants' webhook signing secrets (Step 82): issue, rotate, retire, and open for signing.

A tenant holds at most two secrets, current first. **Rotating** makes a new current one
and keeps the old one as previous, so webhooks are signed with both while the receiver
switches over; issuing again drops the oldest, and `drop_previous` ends the overlap early.
The plaintext is returned once, at issuance (like API keys); stored, it is sealed with
`SecretBox` bound to the tenant id. Step 83 calls `signing_secrets(tenant)` per delivery.
"""

from loguru import logger

from rag_engine.domain import MAX_WEBHOOK_SECRETS, Tenant, WebhookSecret, utc_now
from rag_engine.exceptions import ConflictError
from rag_engine.repositories import TenantRepository
from rag_engine.security import SecretBox
from rag_engine.webhooks.signer import display_prefix, generate_secret

_CAS_ATTEMPTS = 3


class WebhookSecretConflictError(ConflictError):
    code = "webhook_secret_conflict"
    default_message = "The tenant's webhook secrets changed concurrently; retry"


class WebhookSecretService:
    def __init__(self, tenants: TenantRepository, box: SecretBox) -> None:
        self._tenants = tenants
        self._box = box

    async def rotate(self, tenant_id: str) -> tuple[Tenant, str]:
        """A new current secret (plaintext returned once); the old current becomes previous."""
        for _ in range(_CAS_ATTEMPTS):
            tenant = await self._tenants.get_or_raise(tenant_id)
            plaintext = generate_secret()
            fresh = WebhookSecret(
                sealed=self._box.seal(plaintext, context=tenant.id),
                prefix=display_prefix(plaintext),
                created_at=utc_now(),
            )
            kept = tenant.webhook_secrets[: MAX_WEBHOOK_SECRETS - 1]
            updated = await self._tenants.replace_webhook_secrets(tenant, [fresh, *kept])
            if updated is not None:
                logger.bind(tenant_id=tenant_id).info(
                    "issued webhook secret {} ({} active)", fresh.prefix, 1 + len(kept)
                )
                return updated, plaintext
        raise WebhookSecretConflictError(details={"tenant_id": tenant_id})

    async def drop_previous(self, tenant_id: str) -> Tenant:
        """End a rotation's overlap: only the current secret signs from now on."""
        for _ in range(_CAS_ATTEMPTS):
            tenant = await self._tenants.get_or_raise(tenant_id)
            if len(tenant.webhook_secrets) <= 1:
                return tenant
            updated = await self._tenants.replace_webhook_secrets(
                tenant, tenant.webhook_secrets[:1]
            )
            if updated is not None:
                logger.bind(tenant_id=tenant_id).info("retired the previous webhook secret")
                return updated
        raise WebhookSecretConflictError(details={"tenant_id": tenant_id})

    async def signing_secrets(self, tenant: Tenant) -> list[str]:
        """Every active secret in plaintext, current first (empty: none issued).

        Secrets sealed under a retired encryption key are re-sealed under the current one
        on the way (best effort: a concurrent change just postpones it).
        """
        plaintexts = [self._box.open(s.sealed, context=tenant.id) for s in tenant.webhook_secrets]
        if any(self._box.needs_reseal(s.sealed) for s in tenant.webhook_secrets):
            resealed = [
                s.model_copy(update={"sealed": self._box.seal(p, context=tenant.id)})
                for s, p in zip(tenant.webhook_secrets, plaintexts, strict=True)
            ]
            if await self._tenants.replace_webhook_secrets(tenant, resealed) is not None:
                logger.bind(tenant_id=tenant.id).info("re-sealed webhook secrets")
        return plaintexts
