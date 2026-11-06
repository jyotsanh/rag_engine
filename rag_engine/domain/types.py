"""Reusable validated field types for domain models."""

import re
from typing import Annotated
from urllib.parse import urlsplit

from pydantic import AfterValidator, Field, StringConstraints

MAX_URL_LENGTH = 2048
_ID_HEX = r"[0-9a-f]{32}"


def _prefixed(prefix: str) -> StringConstraints:
    """`<prefix>_<32 hex>`, the id format `MongoDocument` generates."""
    return StringConstraints(pattern=rf"^{re.escape(prefix)}_{_ID_HEX}$")


TenantId = Annotated[str, _prefixed("ten")]
ApiKeyId = Annotated[str, _prefixed("key")]
KnowledgeBaseId = Annotated[str, _prefixed("kb")]
SourceId = Annotated[str, _prefixed("src")]
JobId = Annotated[str, _prefixed("job")]
PageId = Annotated[str, _prefixed("page")]
ChunkId = Annotated[str, _prefixed("chk")]


def _validate_http_url(value: str) -> str:
    """Absolute http(s) URL with a host. Kept verbatim; Step 23 normalizes before storage."""
    if len(value) > MAX_URL_LENGTH:
        raise ValueError(f"URL longer than {MAX_URL_LENGTH} characters")
    if any(ch.isspace() for ch in value):
        raise ValueError("URL must not contain whitespace")
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"}:
        raise ValueError("URL scheme must be http or https")
    if not parts.hostname:
        raise ValueError("URL must include a host")
    if parts.username or parts.password:
        raise ValueError("URL must not embed credentials")
    return value


HttpUrlStr = Annotated[str, AfterValidator(_validate_http_url)]

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
Description = Annotated[str, StringConstraints(strip_whitespace=True, max_length=2000)]


def compile_glob(pattern: str) -> re.Pattern[str]:
    """URL-path glob → anchored regex. `**` spans `/`; `*` and `?` stay within a segment."""
    if not pattern.startswith("/"):
        raise ValueError(f"pattern must start with '/': {pattern!r}")
    out: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def _valid_glob(pattern: str) -> str:
    compile_glob(pattern)
    return pattern


def _valid_path_prefix(prefix: str) -> str:
    if not prefix.startswith("/"):
        raise ValueError("path_prefix must start with '/'")
    if any(ch in prefix for ch in "?#*") or any(ch.isspace() for ch in prefix):
        raise ValueError("path_prefix must be a plain path (no query, fragment, globs or spaces)")
    return prefix


# glob over URL paths, e.g. "/docs/**" (see `compile_glob`)
PathPattern = Annotated[
    str, StringConstraints(min_length=1, max_length=500), AfterValidator(_valid_glob)
]
PathPrefix = Annotated[
    str, StringConstraints(min_length=1, max_length=500), AfterValidator(_valid_path_prefix)
]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0)]
