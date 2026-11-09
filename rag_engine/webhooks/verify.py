"""Verifying a rag_engine webhook, for receivers. **Standard library only**: copy this file
into your service as is, or `pip install` rag_engine and import it.

Every webhook carries::

    X-RagEngine-Signature: t=1731150000,v1=5257a869e7ec…(64 hex chars)

- ``t`` is the Unix time the request was signed at;
- each ``v1`` is ``hex(HMAC-SHA256(secret, f"{t}." + raw request body))``. During a secret
  rotation there are two ``v1`` entries, one per active secret; any match is valid.
  Unknown schemes (``v2=…``) are ignored, so the format can evolve.

Verify against the **raw body bytes** (before any JSON parsing), and reject old timestamps
(default tolerance 5 minutes) so a captured request can't be replayed later::

    from verify import verify_webhook, WebhookVerificationError

    try:
        verify_webhook(request.body, request.headers.get("X-RagEngine-Signature"), SECRET)
    except WebhookVerificationError:
        return 400

To be safe against replays within the tolerance as well, remember the
``X-RagEngine-Delivery`` ids you have processed and drop repeats.
"""

import hashlib
import hmac
import time
from collections.abc import Sequence

SIGNATURE_HEADER = "X-RagEngine-Signature"
SCHEME = "v1"
DEFAULT_TOLERANCE_SECONDS = 300
_MAX_HEADER_LENGTH = 1024  # a few signatures; anything longer is not ours
MAX_SIGNATURES = 8


class WebhookVerificationError(Exception):
    """The request is not a valid, fresh webhook from rag_engine."""


class MalformedSignatureError(WebhookVerificationError):
    """The signature header is missing or unparseable."""


class TimestampOutOfToleranceError(WebhookVerificationError):
    """The request was signed too long ago (a replay?) or in the future (clock skew)."""


class SignatureMismatchError(WebhookVerificationError):
    """No signature matches any of the secrets (wrong secret, or a tampered body)."""


def compute_signature(secret: str, timestamp: int, body: bytes) -> str:
    """`hex(HMAC-SHA256(secret, b"{timestamp}." + body))`."""
    message = f"{timestamp}.".encode() + body
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def parse_signature_header(header: str | None) -> tuple[int, list[str]]:
    """`(timestamp, [v1 signatures])`, or `MalformedSignatureError`."""
    if not header:
        raise MalformedSignatureError(f"missing {SIGNATURE_HEADER} header")
    if len(header) > _MAX_HEADER_LENGTH:
        raise MalformedSignatureError("signature header too long")
    timestamp: int | None = None
    signatures: list[str] = []
    for part in header.split(","):
        key, sep, value = part.strip().partition("=")
        if not sep or not value:
            raise MalformedSignatureError(f"malformed signature element {part.strip()!r}")
        if key == "t":
            if timestamp is not None or not value.isdigit():
                raise MalformedSignatureError("invalid or repeated timestamp")
            timestamp = int(value)
        elif key == SCHEME:
            signatures.append(value.lower())
    if timestamp is None:
        raise MalformedSignatureError("no timestamp in the signature header")
    if not signatures:
        raise MalformedSignatureError(f"no {SCHEME} signature in the header")
    if len(signatures) > MAX_SIGNATURES:
        raise MalformedSignatureError("too many signatures")
    return timestamp, signatures


def verify_webhook(
    body: bytes,
    header: str | None,
    secrets: str | Sequence[str],
    *,
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
    now: float | None = None,
) -> int:
    """Check `header` against the raw `body`; returns the signed timestamp.

    `secrets` is your webhook secret, or several while you rotate yours. Raises a
    `WebhookVerificationError` subclass when the request must be rejected.
    """
    if isinstance(secrets, str):
        secrets = [secrets]
    if not secrets or not all(secrets):
        raise ValueError("at least one non-empty secret is required")
    timestamp, signatures = parse_signature_header(header)
    current = time.time() if now is None else now
    if abs(current - timestamp) > tolerance_seconds:
        raise TimestampOutOfToleranceError(
            f"signed at {timestamp}, {current - timestamp:.0f} s from now "
            f"(tolerance {tolerance_seconds:.0f} s)"
        )
    expected = [compute_signature(secret, timestamp, body) for secret in secrets]
    # constant-time comparisons, and no early exit that would leak which one matched
    matched = False
    for candidate in signatures:
        for want in expected:
            matched |= hmac.compare_digest(candidate, want)
    if not matched:
        raise SignatureMismatchError("no signature matches")
    return timestamp
