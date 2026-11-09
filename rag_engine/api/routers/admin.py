"""Operator endpoints: tenants, their API keys and webhook signing secrets. All require the
master admin key."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, Query, Response, status

from rag_engine.api.deps import AdminAuth, ApiKeys, Box, Hasher, Tenants
from rag_engine.api.params import Cursor, Limit
from rag_engine.api.schemas import (
    ApiKeyCreate,
    ApiKeyCreatedResponse,
    ApiKeyResponse,
    TenantCreate,
    TenantPatch,
    TenantResponse,
    WebhookSecretCreatedResponse,
    WebhookSecretInfo,
)
from rag_engine.domain import TenantStatus
from rag_engine.domain.types import ApiKeyId, TenantId
from rag_engine.exceptions import ErrorResponse
from rag_engine.repositories import DEFAULT_PAGE_SIZE, Page
from rag_engine.services import TenantAdminService
from rag_engine.webhooks import WebhookSecretService

_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (401, 403, 404, 409)
}

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[AdminAuth], responses=_ERRORS)


def get_service(tenants: Tenants, keys: ApiKeys, hasher: Hasher) -> TenantAdminService:
    return TenantAdminService(tenants, keys, hasher)


Service = Annotated[TenantAdminService, Depends(get_service)]


def get_webhook_secrets(tenants: Tenants, box: Box) -> WebhookSecretService:
    return WebhookSecretService(tenants, box)


WebhookSecrets = Annotated[WebhookSecretService, Depends(get_webhook_secrets)]
TenantPath = Annotated[TenantId, Path(description="Tenant id, e.g. `ten_…`.")]
KeyPath = Annotated[ApiKeyId, Path(description="API key id, e.g. `key_…`.")]


@router.post("/tenants", status_code=status.HTTP_201_CREATED, summary="Create a tenant")
async def create_tenant(body: TenantCreate, service: Service) -> TenantResponse:
    tenant = await service.create_tenant(
        body.name,
        default_embedding_model=body.default_embedding_model,
        quota_patch=body.quotas.model_dump(exclude_none=True) if body.quotas else {},
    )
    return TenantResponse.of(tenant)


@router.get("/tenants", summary="List tenants (newest first)")
async def list_tenants(
    tenants: Tenants,
    status_filter: Annotated[TenantStatus | None, Query(alias="status")] = None,
    limit: Limit = DEFAULT_PAGE_SIZE,
    cursor: Cursor = None,
) -> Page[TenantResponse]:
    page = await tenants.paginate(
        {"status": status_filter} if status_filter else None, limit=limit, cursor=cursor
    )
    return Page[TenantResponse](
        items=[TenantResponse.of(t) for t in page.items], next_cursor=page.next_cursor
    )


@router.get("/tenants/{tenant_id}", summary="Get a tenant")
async def get_tenant(tenant_id: TenantPath, tenants: Tenants) -> TenantResponse:
    return TenantResponse.of(await tenants.get_or_raise(tenant_id))


@router.patch("/tenants/{tenant_id}", summary="Update a tenant (name, status, model, quotas)")
async def update_tenant(
    tenant_id: TenantPath, body: TenantPatch, service: Service
) -> TenantResponse:
    changes = body.model_dump(exclude_unset=True)
    return TenantResponse.of(await service.update_tenant(tenant_id, changes))


@router.post(
    "/tenants/{tenant_id}/keys",
    status_code=status.HTTP_201_CREATED,
    summary="Issue an API key (the plaintext is returned once)",
)
async def issue_key(
    tenant_id: TenantPath, body: ApiKeyCreate, service: Service, response: Response
) -> ApiKeyCreatedResponse:
    record, generated = await service.issue_key(
        tenant_id, name=body.name, expires_at=body.expires_at
    )
    response.headers["Cache-Control"] = "no-store"  # the body carries a secret
    return ApiKeyCreatedResponse(
        **ApiKeyResponse.of(record).model_dump(), api_key=generated.plaintext.get_secret_value()
    )


@router.get("/tenants/{tenant_id}/keys", summary="List a tenant's API keys")
async def list_keys(
    tenant_id: TenantPath,
    service: Service,
    include_revoked: bool = False,
    limit: Limit = DEFAULT_PAGE_SIZE,
    cursor: Cursor = None,
) -> Page[ApiKeyResponse]:
    page = await service.list_keys(
        tenant_id, include_revoked=include_revoked, limit=limit, cursor=cursor
    )
    return Page[ApiKeyResponse](
        items=[ApiKeyResponse.of(k) for k in page.items], next_cursor=page.next_cursor
    )


@router.delete("/tenants/{tenant_id}/keys/{key_id}", summary="Revoke an API key (idempotent)")
async def revoke_key(tenant_id: TenantPath, key_id: KeyPath, service: Service) -> ApiKeyResponse:
    return ApiKeyResponse.of(await service.revoke_key(tenant_id, key_id))


@router.post(
    "/tenants/{tenant_id}/webhook-secret",
    status_code=status.HTTP_201_CREATED,
    summary="Issue or rotate the webhook signing secret (the plaintext is returned once)",
    description="The new secret becomes current; the previous current one keeps signing "
    "alongside it (two `v1` signatures per webhook) until the next rotation or until it is "
    "retired with `DELETE …/webhook-secret/previous`, so receivers can switch without "
    "downtime.",
)
async def rotate_webhook_secret(
    tenant_id: TenantPath, secrets: WebhookSecrets, response: Response
) -> WebhookSecretCreatedResponse:
    tenant, plaintext = await secrets.rotate(tenant_id)
    response.headers["Cache-Control"] = "no-store"  # the body carries a secret
    active = TenantResponse.of(tenant).webhook_secrets
    return WebhookSecretCreatedResponse(
        secret=plaintext, prefix=active[0].prefix, created_at=active[0].created_at,
        active=active,
    )  # fmt: skip


@router.delete(
    "/tenants/{tenant_id}/webhook-secret/previous",
    summary="Retire the previous webhook secret (idempotent)",
)
async def drop_previous_webhook_secret(
    tenant_id: TenantPath, secrets: WebhookSecrets
) -> list[WebhookSecretInfo]:
    return TenantResponse.of(await secrets.drop_previous(tenant_id)).webhook_secrets
