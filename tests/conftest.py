"""Root test configuration.

- Injects deterministic secrets before `rag_engine.config.settings` is imported.
- Auto-marks tests by directory (`tests/unit` → `unit`, `tests/integration` → `integration`).
- Skips `live` tests unless `--run-live` is given.
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
    "OPENAI_API_KEY": "sk-test-not-a-real-key",
}

for _key, _value in TEST_ENV.items():
    os.environ.setdefault(_key, _value)

from rag_engine.config import Settings, settings  # noqa: E402
from tests.support.fixture_site import FixtureSite  # noqa: E402

pytest_plugins = ["pytester"]

TESTS_ROOT = Path(__file__).parent
_DIRECTORY_MARKERS = {"unit": "unit", "integration": "integration", "browser": "browser"}

OverrideSettings = Callable[..., Settings]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run tests marked `live` (real OpenAI / public websites; costs money)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_live = pytest.mark.skip(reason="live test: pass --run-live to run")
    for item in items:
        path = Path(item.path)
        if path.is_relative_to(TESTS_ROOT):
            marker = _DIRECTORY_MARKERS.get(path.relative_to(TESTS_ROOT).parts[0])
            if marker is not None:
                item.add_marker(marker)
        if "live" in item.keywords and not config.getoption("--run-live"):
            item.add_marker(skip_live)


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
