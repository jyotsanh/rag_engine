"""Root test configuration.

- Injects deterministic secrets before `rag_engine.config.settings` is imported.
- Auto-marks tests by directory (`tests/unit` → `unit`, `tests/integration` → `integration`).
- Auto-marks `tests/browser` → `browser` and `tests/container` → `container`.
- Skips `live` tests unless `--run-live`, `container` tests unless `--run-container`.
- `--update-golden` rewrites golden snapshots (`tests/support/golden.py`) instead of comparing.
- Provides `override_settings` for validated, auto-restored settings changes.
"""

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

TEST_ENV: dict[str, str] = {
    "ENVIRONMENT": "test",
    "ADMIN_API_KEY": "test-admin-key-0123456789abcdef0123456789",
    "API_KEY_PEPPER": "test-pepper-0123456789abcdef0123456789abcd",
    "SECRET_ENCRYPTION_KEY": "test-secret-box-0123456789abcdef01234567",
    "OPENAI_API_KEY": "sk-test-not-a-real-key",
    # public models (fastembed BM25) need no token; an expired local `hf auth login` token
    # would otherwise be sent and rejected with 401
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
}

for _key, _value in TEST_ENV.items():
    os.environ.setdefault(_key, _value)

from rag_engine.config import Settings, settings  # noqa: E402
from tests.support.fixture_site import FixtureSite  # noqa: E402

pytest_plugins = ["pytester", "tests.support.containers"]

TESTS_ROOT = Path(__file__).parent
_DIRECTORY_MARKERS = {
    "unit": "unit",
    "integration": "integration",
    "browser": "browser",
    "container": "container",
}

OverrideSettings = Callable[..., Settings]
# app startup against real services also bootstraps the vector store (Step 74): give the
# lifespan more than asgi-lifespan's 5 s default so a busy Docker host can't fail setup
APP_STARTUP_TIMEOUT = 30


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run tests marked `live` (real OpenAI / public websites; costs money)",
    )
    parser.addoption(
        "--run-container",
        action="store_true",
        default=False,
        help="run tests marked `container` (builds the Docker images; minutes)",
    )
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="rewrite golden snapshot files instead of comparing against them",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_live = pytest.mark.skip(reason="live test: pass --run-live to run")
    skip_container = pytest.mark.skip(reason="container test: pass --run-container to run")
    for item in items:
        path = Path(item.path)
        if path.is_relative_to(TESTS_ROOT):
            marker = _DIRECTORY_MARKERS.get(path.relative_to(TESTS_ROOT).parts[0])
            if marker is not None:
                item.add_marker(marker)
        if "live" in item.keywords and not config.getoption("--run-live"):
            item.add_marker(skip_live)
        if "container" in item.keywords and not config.getoption("--run-container"):
            item.add_marker(skip_container)


@pytest.fixture
def override_settings(monkeypatch: pytest.MonkeyPatch) -> OverrideSettings:
    """Patch fields on the global `settings` singleton for one test.

    Overrides are validated through a full `Settings(...)` build first, so an
    invalid combination fails the test instead of silently applying. Every
    change is reverted by monkeypatch at teardown.
    """

    def _override(**overrides: Any) -> Settings:
        unknown = set(overrides) - set(Settings.model_fields)
        if unknown:
            raise AttributeError(f"unknown settings: {sorted(unknown)}")
        validated = Settings(_env_file=None, **{**settings.model_dump(), **overrides})
        for name in overrides:
            monkeypatch.setattr(settings, name, getattr(validated, name))
        return settings

    return _override


@pytest.fixture(scope="session")
def fixture_site_server() -> Iterator[FixtureSite]:
    site = FixtureSite().start()
    yield site
    site.stop()


@pytest.fixture
def fixture_site(fixture_site_server: FixtureSite) -> FixtureSite:
    """The local edge-case website (see `tests/support/fixture_site.py`), reset per test."""
    fixture_site_server.reset()
    return fixture_site_server
