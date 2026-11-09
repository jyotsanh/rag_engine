"""Signing outgoing webhooks (Step 82). The receiver side is `rag_engine.webhooks.verify`.

`sign(body, secrets, timestamp)` builds the `X-RagEngine-Signature` value
`t=<unix>,v1=<hex>[,v1=<hex>]`: one HMAC-SHA256 per active secret of the tenant (current
first; two while a rotation overlaps), over `f"{t}." + body`. Secrets are per tenant, so
one tenant can never forge a webhook another tenant's receiver would accept.
"""

import secrets as _secrets
import time
from collections.abc import Sequence

from rag_engine.webhooks.verify import (
    MAX_SIGNATURES,
    SCHEME,
    SIGNATURE_HEADER,
    compute_signature,
)

SECRET_PREFIX = "whsec_"
_SECRET_BYTES = 32  # 256 bits, as HMAC-SHA256's block-size-independent strength
DISPLAY_PREFIX_LENGTH = len(SECRET_PREFIX) + 6  # e.g. `whsec_AbC123`, shown in listings


def generate_secret() -> str:
    """A new webhook secret: `whsec_` + 43 URL-safe characters (256 random bits)."""
    return SECRET_PREFIX + _secrets.token_urlsafe(_SECRET_BYTES)


def display_prefix(secret: str) -> str:
    """The non-secret head of a secret, to tell secrets apart in listings."""
    return secret[:DISPLAY_PREFIX_LENGTH]


def sign(body: bytes, secrets: Sequence[str], *, timestamp: int) -> str:
    """The `X-RagEngine-Signature` value for `body`, signed at `timestamp`."""
    if not secrets or not all(secrets):
        raise ValueError("at least one non-empty secret is required")
    if len(secrets) > MAX_SIGNATURES:
        raise ValueError(f"at most {MAX_SIGNATURES} secrets")
    if timestamp < 0:
        raise ValueError("timestamp must be >= 0")
    parts = [f"t={timestamp}"]
    parts += [f"{SCHEME}={compute_signature(s, timestamp, body)}" for s in secrets]
    return ",".join(parts)


def signature_headers(
    body: bytes, secrets: Sequence[str], *, now: float | None = None
) -> dict[str, str]:
    """`{SIGNATURE_HEADER: …}` for a request sent now."""
    timestamp = int(time.time() if now is None else now)
    return {SIGNATURE_HEADER: sign(body, secrets, timestamp=timestamp)}
