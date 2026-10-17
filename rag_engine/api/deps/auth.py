"""Authentication dependencies.

- `require_admin`: the master `ADMIN_API_KEY` (operators, `/admin/*`).
- `require_tenant`: a tenant API key (`Authorization: Bearer rk_...`).

Every authentication failure returns the same 401 body, so a caller cannot tell
a malformed key from an unknown, wrong, revoked or expired one. A *valid* key
whose tenant is disabled gets 403, since its holder has already proven identity.
"""

import hashlib
import hmac
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from rag_engine.api.deps.infrastructure import ApiKeys, Hasher, Tenants
from rag_engine.config import bind_log_context, logger, settings
from rag_engine.domain import ApiKey, Tenant
from rag_engine.exceptions import ForbiddenError, TooManyRequestsError, UnauthorizedError
from rag_engine.infrastructure import RedisNamespace
from rag_engine.security import SlidingWindowRateLimiter, parse_api_key

API_RATE_LIMIT_WINDOW_SECONDS = 60


class InvalidCredentialsError(UnauthorizedError):
    code = "invalid_credentials"
    default_message = "Missing or invalid API key"


class TenantDisabledError(ForbiddenError):
    code = "tenant_disabled"
    default_message = "This tenant is disabled"


# auto_error=False: we raise our own uniform error instead of FastAPI's 403
_admin_scheme = HTTPBearer(
    scheme_name="AdminKey",
    description="Master admin key (`ADMIN_API_KEY`) for `/admin/*` endpoints.",
    auto_error=False,
)
_tenant_scheme = HTTPBearer(
    scheme_name="TenantApiKey",
    description="Tenant API key: `Authorization: Bearer rk_<prefix>_<secret>`.",
    auto_error=False,
)


@dataclass(frozen=True, slots=True)
class TenantContext:
    tenant: Tenant
    api_key: ApiKey

    @property
    def tenant_id(self) -> str:
        return self.tenant.id

    @property
    def scope(self) -> dict[str, str]:
        """Repository `scope=` filter that confines queries to this tenant."""
        return {"tenant_id": self.tenant.id}


def _token(credentials: HTTPAuthorizationCredentials | None) -> str | None:
    return credentials.credentials if credentials is not None else None


def _digest(value: str) -> bytes:
    # compare fixed-length digests so the comparison time does not depend on input length
    return hashlib.sha256(value.encode()).digest()


async def require_admin(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_admin_scheme)],
) -> None:
    token = _token(credentials) or ""
    expected = settings.ADMIN_API_KEY.get_secret_value()
    if not hmac.compare_digest(_digest(token), _digest(expected)) or not token:
        logger.warning("rejected admin request: invalid admin key")
        raise InvalidCredentialsError()


async def require_tenant(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_tenant_scheme)],
    keys: ApiKeys,
    tenants: Tenants,
    hasher: Hasher,
) -> TenantContext:
    parsed = parse_api_key(_token(credentials))
    if parsed is None:
        raise InvalidCredentialsError()

    record = await keys.get_by_prefix(parsed.prefix)
    if record is None:
        hasher.burn(parsed.plaintext)  # same work as a real verify: no timing oracle
        raise InvalidCredentialsError()

    result = hasher.verify(parsed.plaintext, record.key_hash)
    if not result.valid:
        logger.bind(key_id=record.id).warning("rejected API key: wrong secret")
        raise InvalidCredentialsError()
    if not record.is_usable():
        logger.bind(key_id=record.id).info("rejected API key: revoked or expired")
        raise InvalidCredentialsError()

    tenant = await tenants.get(record.tenant_id)
    if tenant is None:
        raise InvalidCredentialsError()
    bind_log_context(tenant_id=tenant.id)
    if not tenant.is_active:
        raise TenantDisabledError()

    if result.needs_rehash:
        await keys.rehash(record.id, parsed.plaintext, hasher)
        logger.bind(key_id=record.id).info("re-hashed API key with the current pepper")
    await keys.touch_last_used(record.id)
    return TenantContext(tenant=tenant, api_key=record)


async def require_tenant_within_rate_limit(
    ctx: Annotated[TenantContext, Depends(require_tenant)],
    request: Request,
    response: Response,
) -> TenantContext:
    """Authenticated tenant, counted against `quotas.api_rate_limit_per_minute`.

    Fails open if Redis is unavailable: readiness already reports the outage, and
    the limiter must not turn a Redis blip into a full API outage.
    """
    redis = request.app.state.infrastructure.redis
    limiter: SlidingWindowRateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    try:
        if limiter is None:
            limiter = request.app.state.rate_limiter = SlidingWindowRateLimiter(redis.client)
        decision = await limiter.hit(
            redis.keys.key(RedisNamespace.API_RATE_LIMIT, ctx.tenant_id),
            limit=ctx.tenant.quotas.api_rate_limit_per_minute,
            window_seconds=API_RATE_LIMIT_WINDOW_SECONDS,
        )
    except Exception:
        logger.opt(exception=True).error("rate limiter unavailable; allowing request")
        return ctx
    if not decision.allowed:
        logger.info("tenant rate limit exceeded")
        raise TooManyRequestsError(
            "API rate limit exceeded",
            retry_after=decision.retry_after,
            details={"limit": decision.limit, "window_seconds": decision.window_seconds},
            headers=decision.headers(),
        )
    response.headers.update(decision.headers())
    return ctx


AdminAuth = Depends(require_admin)
# every tenant route uses this: authentication + per-tenant rate limit
CurrentTenant = Annotated[TenantContext, Depends(require_tenant_within_rate_limit)]
