"""Tenant API key generation, hashing and verification.

Key format: `rk_<prefix>_<secret>`
- `prefix`: 12 base62 chars, public, stored in clear, used to look the key up.
- `secret`: 43 base62 chars (~256 bits), never stored.

Only `HMAC-SHA256(pepper, full key)` is persisted. The pepper lives in settings,
not the database, so a database leak alone cannot be brute-forced offline.
"""

import hashlib
import hmac
import re
import secrets
import string
from dataclasses import dataclass, field

from pydantic import SecretStr

from rag_engine.config import Settings, settings

KEY_SCHEME = "rk"
PREFIX_LENGTH = 12
SECRET_LENGTH = 43
_ALPHABET = string.ascii_letters + string.digits
# accept the range the ApiKey model allows, so older/newer generations still parse
_KEY_PATTERN = re.compile(rf"^{KEY_SCHEME}_([A-Za-z0-9]{{8,16}})_([A-Za-z0-9]{{32,64}})$")


@dataclass(frozen=True, slots=True)
class GeneratedApiKey:
    """A new key. `plaintext` is shown to the caller exactly once and then discarded."""

    plaintext: SecretStr
    prefix: str
    key_hash: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class ParsedApiKey:
    prefix: str
    plaintext: SecretStr


@dataclass(frozen=True, slots=True)
class VerificationResult:
    valid: bool
    # matched a retired pepper: the caller should store `rehash()` of the key
    needs_rehash: bool = False


def _random(length: int) -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(length))


def parse_api_key(raw: str | None) -> ParsedApiKey | None:
    """Split a presented key into its lookup prefix; `None` if it is not a well-formed key."""
    if not raw:
        return None
    match = _KEY_PATTERN.fullmatch(raw.strip())
    if match is None:
        return None
    return ParsedApiKey(prefix=match.group(1), plaintext=SecretStr(match.group(0)))


def display_hint(prefix: str) -> str:
    """Safe-to-show identifier for listings, e.g. `rk_AbCdEf123456_…`."""
    return f"{KEY_SCHEME}_{prefix}_…"


class ApiKeyHasher:
    def __init__(self, config: Settings = settings) -> None:
        self._current = config.API_KEY_PEPPER.get_secret_value().encode()
        self._previous = [p.get_secret_value().encode() for p in config.API_KEY_PEPPER_PREVIOUS]

    @staticmethod
    def _hmac(pepper: bytes, plaintext: str) -> str:
        return hmac.new(pepper, plaintext.encode(), hashlib.sha256).hexdigest()

    def hash(self, plaintext: SecretStr) -> str:
        return self._hmac(self._current, plaintext.get_secret_value())

    def generate(self) -> GeneratedApiKey:
        prefix = _random(PREFIX_LENGTH)
        plaintext = SecretStr(f"{KEY_SCHEME}_{prefix}_{_random(SECRET_LENGTH)}")
        return GeneratedApiKey(plaintext=plaintext, prefix=prefix, key_hash=self.hash(plaintext))

    def verify(self, plaintext: SecretStr, stored_hash: str) -> VerificationResult:
        """Constant-time comparison against the current pepper, then any retired ones.

        Every pepper is always checked, so timing does not reveal which one matched.
        """
        key = plaintext.get_secret_value()
        current = hmac.compare_digest(self._hmac(self._current, key), stored_hash)
        previous = False
        for pepper in self._previous:
            previous |= hmac.compare_digest(self._hmac(pepper, key), stored_hash)
        if current:
            return VerificationResult(valid=True)
        return VerificationResult(valid=previous, needs_rehash=previous)

    def burn(self, plaintext: SecretStr | None = None) -> None:
        """Do the same hashing work as `verify` when no key was found for a prefix,
        so 'unknown prefix' and 'wrong secret' take the same time."""
        self.verify(plaintext or SecretStr(""), "0" * 64)
