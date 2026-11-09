"""Encrypting secrets the service must read back (Step 82: tenants' webhook secrets).

Unlike API keys (hashed: only verified), a webhook secret is needed in plaintext to sign
with, so it is stored **encrypted**: AES-256-GCM under a key derived from
`SECRET_ENCRYPTION_KEY`, with the owning record's id as associated data (a sealed value
copied onto another tenant doesn't open there).

Token format: `sb1.<key id>.<base64url(nonce ‖ ciphertext+tag)>`. The key id (first 8 hex
of the derived key's hash) picks the key on `open()`, so keys rotate: put the new one in
`SECRET_ENCRYPTION_KEY`, the old one in `SECRET_ENCRYPTION_KEYS_PREVIOUS`, and anything
sealed under an old key still opens; `needs_reseal()` tells which values to re-encrypt.
"""

import base64
import hashlib
import os
from collections.abc import Sequence

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import SecretStr

from rag_engine.config import Settings, settings
from rag_engine.exceptions import AppBaseException

_VERSION = "sb1"
_NONCE_BYTES = 12
_KDF_CONTEXT = b"rag_engine/secret-box/v1\x00"


class SecretUnreadableError(AppBaseException):
    """A sealed secret can't be opened: unknown key (rotated away?), or tampered data."""

    code = "secret_unreadable"
    default_message = "A stored secret could not be decrypted"


class SecretBox:
    def __init__(self, keys: Sequence[SecretStr]) -> None:
        """`keys`: the current key first, then retired ones still accepted by `open()`."""
        if not keys:
            raise ValueError("at least one encryption key is required")
        self._keys: dict[str, AESGCM] = {}
        self._current: str = ""
        for n, key in enumerate(keys):
            derived = hashlib.sha256(_KDF_CONTEXT + key.get_secret_value().encode()).digest()
            key_id = hashlib.sha256(derived).hexdigest()[:8]
            self._keys.setdefault(key_id, AESGCM(derived))
            if n == 0:
                self._current = key_id

    @classmethod
    def from_settings(cls, config: Settings = settings) -> SecretBox:
        return cls([config.SECRET_ENCRYPTION_KEY, *config.SECRET_ENCRYPTION_KEYS_PREVIOUS])

    def seal(self, plaintext: str, *, context: str) -> str:
        nonce = os.urandom(_NONCE_BYTES)
        sealed = self._keys[self._current].encrypt(nonce, plaintext.encode(), context.encode())
        payload = base64.urlsafe_b64encode(nonce + sealed).decode().rstrip("=")
        return f"{_VERSION}.{self._current}.{payload}"

    def open(self, token: str, *, context: str) -> str:
        version, key_id, payload = self._split(token)
        aead = self._keys.get(key_id)
        if version != _VERSION or aead is None:
            raise SecretUnreadableError(details={"key_id": key_id, "version": version})
        try:
            raw = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
            plaintext = aead.decrypt(raw[:_NONCE_BYTES], raw[_NONCE_BYTES:], context.encode())
        except (InvalidTag, ValueError) as exc:
            raise SecretUnreadableError(details={"key_id": key_id}) from exc
        return plaintext.decode()

    def needs_reseal(self, token: str) -> bool:
        """Sealed under a retired key: re-`seal` it under the current one."""
        return self._split(token)[1] != self._current

    @staticmethod
    def _split(token: str) -> tuple[str, str, str]:
        parts = token.split(".")
        if len(parts) != 3:
            raise SecretUnreadableError(details={"reason": "malformed token"})
        return parts[0], parts[1], parts[2]
