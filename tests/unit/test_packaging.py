"""Packaging: every third-party import in `rag_engine` is a runtime dependency."""

import ast
import re
import shutil
import subprocess
import sys
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "rag_engine"
REPO_ROOT = PACKAGE_ROOT.parent


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _third_party_imports() -> dict[str, set[Path]]:
    found: dict[str, set[Path]] = {}
    for path in PACKAGE_ROOT.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                top = name.split(".")[0]
                if top != "rag_engine" and top not in sys.stdlib_module_names:
                    found.setdefault(top, set()).add(path.relative_to(REPO_ROOT))
    return found


def _runtime_distributions() -> set[str]:
    """The locked runtime closure (no dev group), exactly what the images install."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not on PATH")
    exported = subprocess.run(
        [uv, "export", "--frozen", "--no-dev", "--no-hashes", "--no-emit-project", "--quiet"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {
        _normalize(re.split(r"[=<>~!\[; ]", line.strip(), maxsplit=1)[0])
        for line in exported.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "-"))
    }


# ---- #
# Tests
# ---- #


class TestRuntimeDependencies_S37:
    """S37 — Packaging: imports resolve in a `--no-dev` install (the worker image caught
    httpx/httpcore/qdrant-client being dev-only)."""

    def test_every_import_is_a_runtime_dependency(self) -> None:
        runtime = _runtime_distributions()
        providers = packages_distributions()

        missing = {
            module: sorted(str(p) for p in paths)
            for module, paths in _third_party_imports().items()
            if not {_normalize(d) for d in providers.get(module, [])} & runtime
        }

        assert missing == {}, f"imported but not runtime dependencies: {missing}"

    def test_dev_only_tools_are_not_runtime_dependencies(self) -> None:
        runtime = _runtime_distributions()

        assert not {"pytest", "ruff", "ty", "testcontainers", "trustme"} & runtime
