"""Admin-key auth and OpenAPI security schemes (no database needed)."""

from collections.abc import AsyncIterator

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import APIRouter

from rag_engine.api import create_app
from rag_engine.api.deps import AdminAuth, CurrentTenant
from rag_engine.config import settings
from tests.support.fakes import NoopInfrastructure

ADMIN_KEY = settings.ADMIN_API_KEY.get_secret_value()


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(NoopInfrastructure)
    router = APIRouter()

    @router.get("/_test/admin", dependencies=[AdminAuth])
    async def admin_only() -> dict[str, bool]:
        return {"ok": True}

    @router.get("/_test/tenant")
    async def tenant_only(ctx: CurrentTenant) -> dict[str, str]:
        return {"tenant_id": ctx.tenant_id}

    app.include_router(router)
    transport = httpx.ASGITransport(app=app)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=transport, base_url="http://t") as c,
    ):
        yield c


# ---- #
# Tests
# ---- #


class TestAdminAuth_S17:
    """S17 — Auth dependencies: master admin key and documented security schemes."""

    async def test_correct_admin_key_is_accepted(self, client: httpx.AsyncClient) -> None:
        response = await client.get(
            "/_test/admin", headers={"Authorization": f"Bearer {ADMIN_KEY}"}
        )

        assert response.status_code == 200

    @pytest.mark.parametrize(
        "headers",
        [
            {},
            {"Authorization": "Bearer "},
            {"Authorization": "Bearer wrong-key"},
            {"Authorization": f"Bearer {ADMIN_KEY}x"},
            {"Authorization": f"Bearer {ADMIN_KEY[:-1]}"},
            {"Authorization": f"Basic {ADMIN_KEY}"},
            {"Authorization": ADMIN_KEY},
            {"Authorization": "Bearer rk_AbCdEf123456_" + "x" * 43},
            {"X-API-Key": ADMIN_KEY},
        ],
    )
    async def test_anything_else_gets_the_uniform_401(
        self, client: httpx.AsyncClient, headers: dict[str, str]
    ) -> None:
        response = await client.get("/_test/admin", headers=headers)

        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        body = response.json()["error"]
        assert (body["code"], body["message"]) == (
            "invalid_credentials",
            "Missing or invalid API key",
        )
        assert ADMIN_KEY not in response.text

    async def test_openapi_declares_both_security_schemes(self, client: httpx.AsyncClient) -> None:
        spec = (await client.get(f"{settings.API_V1_STR}/openapi.json")).json()

        schemes = spec["components"]["securitySchemes"]
        assert schemes["AdminKey"] == schemes["AdminKey"] | {"type": "http", "scheme": "bearer"}
        assert schemes["TenantApiKey"]["scheme"] == "bearer"
        assert spec["paths"]["/_test/admin"]["get"]["security"] == [{"AdminKey": []}]
        assert spec["paths"]["/_test/tenant"]["get"]["security"] == [{"TenantApiKey": []}]
