"""Test infrastructure: directory markers, live gating, settings overrides."""

import pytest
from pydantic import ValidationError

from rag_engine.config import settings
from tests.conftest import OverrideSettings

_INNER_CONFTEST = """
from tests.conftest import pytest_addoption, pytest_collection_modifyitems  # noqa: F401
"""

_INNER_TESTS = """
import pytest

def test_plain():
    pass

@pytest.mark.live
def test_costs_money():
    pass
"""

_INNER_ARGS = ("-p", "no:cacheprovider", "-o", "asyncio_default_fixture_loop_scope=function")

# ---- #
# Tests
# ---- #


class TestTestInfrastructure_S8:
    """S8 — Test infrastructure: auto markers, `--run-live` gate, `override_settings`."""

    def test_unit_marker_auto_applied(self, request: pytest.FixtureRequest) -> None:
        assert request.node.get_closest_marker("unit") is not None
        assert request.node.get_closest_marker("integration") is None

    def test_live_tests_skipped_by_default(self, pytester: pytest.Pytester) -> None:
        pytester.makeconftest(_INNER_CONFTEST)
        pytester.makepyfile(_INNER_TESTS)

        result = pytester.runpytest_inprocess(*_INNER_ARGS)

        result.assert_outcomes(passed=1, skipped=1)

    def test_run_live_flag_enables_live_tests(self, pytester: pytest.Pytester) -> None:
        pytester.makeconftest(_INNER_CONFTEST)
        pytester.makepyfile(_INNER_TESTS)

        result = pytester.runpytest_inprocess(*_INNER_ARGS, "--run-live")

        result.assert_outcomes(passed=2)

    def test_override_settings_applies_values(self, override_settings: OverrideSettings) -> None:
        override_settings(CRAWL_CONCURRENCY=3, MONGO_DB="other")

        assert settings.CRAWL_CONCURRENCY == 3
        assert settings.MONGO_DB == "other"

    def test_override_settings_is_restored_after_test(self) -> None:
        # runs after the test above (file order): its overrides must be gone
        assert settings.CRAWL_CONCURRENCY == 8
        assert settings.MONGO_DB == "rag_engine"

    def test_override_settings_validates_values(self, override_settings: OverrideSettings) -> None:
        with pytest.raises(ValidationError):
            override_settings(CRAWL_DEFAULT_MAX_PAGES=10**9)

        assert settings.CRAWL_DEFAULT_MAX_PAGES == 500

    def test_override_settings_coerces_types(self, override_settings: OverrideSettings) -> None:
        override_settings(REDIS_PASSWORD="plain", ENVIRONMENT="prod")

        assert settings.REDIS_PASSWORD is not None
        assert settings.REDIS_PASSWORD.get_secret_value() == "plain"
        assert settings.ENVIRONMENT == "prod"

    def test_override_settings_rejects_unknown_fields(
        self, override_settings: OverrideSettings
    ) -> None:
        with pytest.raises(AttributeError, match="NOT_A_SETTING"):
            override_settings(NOT_A_SETTING=1)
