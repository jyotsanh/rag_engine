"""Project scaffold: package import, console scripts, server entrypoints."""

from importlib.metadata import entry_points
from typing import Any

import pytest

import rag_engine
from rag_engine.config import settings

# ---- #
# Tests
# ---- #


class TestUvScaffold_S2:
    """S2 — uv scaffold: package is installed and `dev`/`server` scripts start uvicorn."""

    def test_package_imports_with_public_entrypoints(self) -> None:
        assert callable(rag_engine.dev)
        assert callable(rag_engine.server)
        assert rag_engine.APP_PATH == "rag_engine.api.app:app"

    @pytest.mark.parametrize("name", ["dev", "server"])
    def test_console_script_registered_to_package_function(self, name: str) -> None:
        scripts = {ep.name: ep.value for ep in entry_points(group="console_scripts")}

        assert scripts.get(name) == f"rag_engine:{name}"

    def test_dev_runs_uvicorn_with_reload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        monkeypatch.setattr(rag_engine.uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))

        # Act
        rag_engine.dev()

        # Assert
        assert calls == [
            (
                (rag_engine.APP_PATH,),
                {"host": settings.DEV_HOST, "port": settings.DEV_PORT, "reload": True},
            )
        ]

    def test_server_runs_uvicorn_without_reload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        monkeypatch.setattr(rag_engine.uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))

        rag_engine.server()

        assert calls == [
            ((rag_engine.APP_PATH,), {"host": settings.PROD_HOST, "port": settings.PROD_PORT})
        ]
