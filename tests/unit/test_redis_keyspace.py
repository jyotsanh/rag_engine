"""Redis keyspace rules and AppInfrastructure start/stop ordering (no Redis needed)."""

import pytest
from pydantic import ValidationError

from rag_engine.config import Settings
from rag_engine.infrastructure import AppInfrastructure, RedisKeyspace, RedisNamespace
from rag_engine.infrastructure.exceptions import RedisUnavailableError


class _FakeService:
    def __init__(self, name: str, calls: list[str], fail_on_connect: bool = False) -> None:
        self.name, self.calls, self.fail_on_connect = name, calls, fail_on_connect

    async def connect(self) -> None:
        self.calls.append(f"{self.name}.connect")
        if self.fail_on_connect:
            raise RedisUnavailableError()

    async def ensure_indexes(self) -> None:
        self.calls.append(f"{self.name}.ensure_indexes")

    async def close(self) -> None:
        self.calls.append(f"{self.name}.close")


# ---- #
# Tests
# ---- #


class TestRedisKeyspace_S11:
    """S11 — Redis infrastructure: namespaced key builder and lifecycle ordering."""

    def test_key_joins_prefix_namespace_and_parts(self) -> None:
        keys = RedisKeyspace("rag")

        assert keys.key(RedisNamespace.JOB_CANCEL, "job_123") == "rag:job_cancel:job_123"
        assert keys.key(RedisNamespace.API_RATE_LIMIT, "ten_1", "60") == "rag:api_rl:ten_1:60"

    @pytest.mark.parametrize("part", ["", "has space", "tab\there", "new\nline", "\x00"])
    def test_unsafe_parts_rejected(self, part: str) -> None:
        with pytest.raises(ValueError, match="invalid redis key part"):
            RedisKeyspace("rag").key(RedisNamespace.LOCK, part)

    def test_separator_in_part_rejected_with_hint(self) -> None:
        with pytest.raises(ValueError, match="digest"):
            RedisKeyspace("rag").key(RedisNamespace.ROBOTS, "example.com:443")

    def test_key_requires_a_part(self) -> None:
        with pytest.raises(ValueError, match="at least one part"):
            RedisKeyspace("rag").key(RedisNamespace.LOCK)

    def test_digest_is_stable_short_and_key_safe(self) -> None:
        keys = RedisKeyspace("rag")
        origin = "https://example.com:8443"

        token = keys.digest(origin)

        assert token == keys.digest(origin)
        assert token != keys.digest("https://example.com")
        assert len(token) == 32
        assert keys.key(RedisNamespace.ROBOTS, token).count(":") == 2

    def test_pattern_covers_only_one_namespace(self) -> None:
        assert RedisKeyspace("rag").pattern(RedisNamespace.ROBOTS) == "rag:robots:*"

    @pytest.mark.parametrize("prefix", ["", "Has-Upper", "with:colon", "x" * 33])
    def test_prefix_setting_is_validated(self, prefix: str) -> None:
        with pytest.raises(ValidationError):
            Settings(
                _env_file=None,
                REDIS_KEY_PREFIX=prefix,
                ADMIN_API_KEY="a" * 32,
                API_KEY_PEPPER="p" * 32,
                OPENAI_API_KEY="sk",
            )

    async def test_infrastructure_starts_in_order_and_stops_in_reverse(self) -> None:
        calls: list[str] = []
        infra = AppInfrastructure(
            mongo=_FakeService("mongo", calls),  # type: ignore
            redis=_FakeService("redis", calls),  # type: ignore
        )

        await infra.start()
        await infra.stop()

        assert calls == [
            "mongo.connect",
            "mongo.ensure_indexes",
            "redis.connect",
            "redis.close",
            "mongo.close",
        ]

    async def test_failed_start_closes_already_opened_services(self) -> None:
        calls: list[str] = []
        infra = AppInfrastructure(
            mongo=_FakeService("mongo", calls),  # type: ignore
            redis=_FakeService("redis", calls, fail_on_connect=True),  # type: ignore
        )

        with pytest.raises(RedisUnavailableError):
            await infra.start()

        assert calls[-2:] == ["redis.close", "mongo.close"]
