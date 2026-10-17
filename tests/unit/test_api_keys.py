"""API key service: generation format, peppered HMAC, verification, pepper rotation."""

import hashlib
import hmac
import re

import pytest
from pydantic import SecretStr, ValidationError

from rag_engine.config import Settings
from rag_engine.domain import ApiKey, Tenant
from rag_engine.security import (
    PREFIX_LENGTH,
    SECRET_LENGTH,
    ApiKeyHasher,
    display_hint,
    parse_api_key,
)

OLD_PEPPER = "old-pepper-0123456789abcdef0123456789abcdef"
NEW_PEPPER = "new-pepper-0123456789abcdef0123456789abcdef"


def _config(pepper: str = NEW_PEPPER, previous: list[str] | None = None) -> Settings:
    return Settings(
        _env_file=None,
        ADMIN_API_KEY="a" * 32,
        API_KEY_PEPPER=pepper,
        API_KEY_PEPPER_PREVIOUS=previous or [],
        OPENAI_API_KEY="sk",
    )


@pytest.fixture
def hasher() -> ApiKeyHasher:
    return ApiKeyHasher(_config())


# ---- #
# Tests
# ---- #


class TestApiKeyService_S15:
    """S15 — API key service: rk_<prefix>_<secret>, HMAC-SHA256 + pepper, constant-time verify."""

    def test_generated_key_format(self, hasher: ApiKeyHasher) -> None:
        key = hasher.generate()

        plaintext = key.plaintext.get_secret_value()
        assert re.fullmatch(
            rf"rk_[A-Za-z0-9]{{{PREFIX_LENGTH}}}_[A-Za-z0-9]{{{SECRET_LENGTH}}}", plaintext
        )
        assert plaintext.startswith(f"rk_{key.prefix}_")

    def test_generated_keys_are_unique(self, hasher: ApiKeyHasher) -> None:
        keys = [hasher.generate() for _ in range(500)]

        assert len({k.prefix for k in keys}) == 500
        assert len({k.plaintext.get_secret_value() for k in keys}) == 500

    def test_plaintext_and_hash_hidden_from_repr(self, hasher: ApiKeyHasher) -> None:
        key = hasher.generate()

        assert key.plaintext.get_secret_value() not in repr(key)
        assert key.key_hash not in repr(key)

    def test_hash_is_hmac_sha256_of_full_key_with_pepper(self, hasher: ApiKeyHasher) -> None:
        key = hasher.generate()

        expected = hmac.new(
            NEW_PEPPER.encode(), key.plaintext.get_secret_value().encode(), hashlib.sha256
        ).hexdigest()
        assert key.key_hash == expected

    def test_hash_depends_on_pepper(self) -> None:
        plaintext = SecretStr("rk_AbCdEf123456_" + "x" * 43)

        assert ApiKeyHasher(_config(OLD_PEPPER)).hash(plaintext) != ApiKeyHasher(
            _config(NEW_PEPPER)
        ).hash(plaintext)

    def test_generated_key_fits_the_api_key_model(self, hasher: ApiKeyHasher) -> None:
        key = hasher.generate()

        stored = ApiKey(
            tenant_id=Tenant(name="Acme").id, key_prefix=key.prefix, key_hash=key.key_hash
        )

        assert stored.key_prefix == key.prefix

    def test_verify_accepts_the_right_key(self, hasher: ApiKeyHasher) -> None:
        key = hasher.generate()

        result = hasher.verify(key.plaintext, key.key_hash)

        assert result.valid and not result.needs_rehash

    def test_verify_rejects_a_wrong_secret(self, hasher: ApiKeyHasher) -> None:
        key = hasher.generate()
        tampered = key.plaintext.get_secret_value()[:-1] + (
            "A" if not key.plaintext.get_secret_value().endswith("A") else "B"
        )

        assert not hasher.verify(SecretStr(tampered), key.key_hash).valid
        assert not hasher.verify(hasher.generate().plaintext, key.key_hash).valid

    def test_rotation_accepts_old_pepper_and_flags_rehash(self) -> None:
        old_key = ApiKeyHasher(_config(OLD_PEPPER)).generate()
        rotated = ApiKeyHasher(_config(NEW_PEPPER, previous=[OLD_PEPPER]))

        result = rotated.verify(old_key.plaintext, old_key.key_hash)
        new_hash = rotated.hash(old_key.plaintext)

        assert result.valid and result.needs_rehash
        assert rotated.verify(old_key.plaintext, new_hash) == result.__class__(valid=True)

    def test_without_previous_pepper_old_keys_stop_working(self) -> None:
        old_key = ApiKeyHasher(_config(OLD_PEPPER)).generate()

        assert (
            not ApiKeyHasher(_config(NEW_PEPPER)).verify(old_key.plaintext, old_key.key_hash).valid
        )

    def test_previous_peppers_validated(self) -> None:
        with pytest.raises(ValidationError, match="API_KEY_PEPPER_PREVIOUS"):
            _config(previous=["too-short"])

    def test_burn_never_matches(self, hasher: ApiKeyHasher) -> None:
        hasher.burn(hasher.generate().plaintext)
        hasher.burn()

    def test_parse_extracts_prefix(self, hasher: ApiKeyHasher) -> None:
        key = hasher.generate()

        parsed = parse_api_key(f"  {key.plaintext.get_secret_value()}\n")

        assert parsed is not None
        assert parsed.prefix == key.prefix
        assert parsed.plaintext == key.plaintext

    @pytest.mark.parametrize(
        "raw",
        [
            None,
            "",
            "rk_short_" + "x" * 43,
            "rk_AbCdEf123456_" + "x" * 31,
            "rk_AbCdEf123456_" + "x" * 65,
            "sk_AbCdEf123456_" + "x" * 43,
            "rk_AbCd-f123456_" + "x" * 43,
            "rk_AbCdEf123456_" + "x" * 42 + "!",
            "Bearer rk_AbCdEf123456_" + "x" * 43,
            "rk_AbCdEf123456_" + "x" * 43 + "_extra",
        ],
    )
    def test_parse_rejects_malformed_keys(self, raw: str | None) -> None:
        assert parse_api_key(raw) is None

    def test_display_hint_never_contains_the_secret(self, hasher: ApiKeyHasher) -> None:
        key = hasher.generate()

        hint = display_hint(key.prefix)

        assert hint == f"rk_{key.prefix}_…"
        assert key.plaintext.get_secret_value() not in hint
