"""Canonical URL form used for frontier dedupe, source identity and `url_hash`.

Rules (applied in order):
1. Resolve against `base` if given; strip surrounding whitespace and remove
   tab/CR/LF inside the URL (as browsers do).
2. Scheme: lowercase; only `http` / `https`.
3. Credentials (`user:pass@`) are rejected, never silently dropped.
4. Host: lowercase, IDNA 2008 / UTS #46 to ASCII (`bücher.example` → `xn--bcher-kva.example`),
   trailing dot removed; IPv6 literals stay bracketed and lowercase.
5. Port: default ports (80 for http, 443 for https) removed; invalid ports rejected.
6. Path: dot-segments removed (RFC 3986 §5.2.4); empty path → `/`; percent-encoding
   normalized (hex uppercased, unreserved characters decoded, unsafe characters and
   non-ASCII encoded as UTF-8). Trailing slashes and repeated slashes are preserved:
   servers may serve different resources at `/docs` and `/docs/`.
7. Query: known tracking parameters removed; remaining pairs sorted by key then value
   (blank values kept); `?` dropped when nothing remains.
8. Fragment: dropped.
"""

import hashlib
import re
from urllib.parse import parse_qsl, urljoin, urlsplit, urlunsplit

import idna

DEFAULT_PORTS = {"http": 80, "https": 443}

# exact names, plus any parameter starting with one of TRACKING_PREFIXES
TRACKING_PARAMS = frozenset(
    {
        "gclid", "gclsrc", "dclid", "gbraid", "wbraid", "fbclid", "msclkid", "yclid",
        "twclid", "ttclid", "li_fat_id", "igshid", "mc_cid", "mc_eid", "_ga", "_gl",
        "_hsenc", "_hsmi", "__hstc", "__hssc", "__hsfp", "mkt_tok", "oly_anon_id",
        "oly_enc_id", "vero_id", "rb_clickid", "s_cid",
    }
)  # fmt: skip
TRACKING_PREFIXES = ("utm_", "pk_", "mtm_")

_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
_SUB_DELIMS = frozenset("!$&'()*+,;=")
_PATH_SAFE = _UNRESERVED | _SUB_DELIMS | frozenset(":@/")
# inside a query key/value: `&` and `=` are structural, so they stay encoded
_QUERY_COMPONENT_SAFE = (_UNRESERVED | _SUB_DELIMS | frozenset(":@/?")) - frozenset("&=+")
_INTERNAL_WHITESPACE = re.compile(r"[\t\r\n]")
_PERCENT = re.compile(r"%([0-9A-Fa-f]{2})")


class UrlNormalizationError(ValueError):
    """The input cannot be turned into a crawlable http(s) URL."""


def _percent_encode(text: str, safe: frozenset[str]) -> str:
    """Normalize existing escapes and encode anything outside `safe` as UTF-8."""
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "%" and _PERCENT.match(text, i):
            byte = int(text[i + 1 : i + 3], 16)
            decoded = chr(byte)
            out.append(decoded if decoded in _UNRESERVED else f"%{byte:02X}")
            i += 3
            continue
        if ch in safe:
            out.append(ch)
        else:
            out.extend(f"%{b:02X}" for b in ch.encode("utf-8"))
        i += 1
    return "".join(out)


def _remove_dot_segments(path: str) -> str:
    """RFC 3986 §5.2.4."""
    output: list[str] = []
    segments = path.split("/")
    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        if segment == ".":
            if last:
                output.append("")
            continue
        if segment == "..":
            if len(output) > 1:
                output.pop()
            if last:
                output.append("")
            continue
        output.append(segment)
    result = "/".join(output)
    if path.startswith("/") and not result.startswith("/"):
        result = "/" + result
    return result


def _normalize_host(hostname: str) -> str:
    host = hostname.rstrip(".").lower()
    if not host:
        raise UrlNormalizationError("URL has no host")
    if ":" in host:  # IPv6 literal (urlsplit already removed the brackets)
        return f"[{host}]"
    try:
        return idna.encode(host, uts46=True).decode("ascii")
    except idna.IDNAError as exc:
        raise UrlNormalizationError(f"invalid host {hostname!r}: {exc}") from exc


def _is_tracking(name: str) -> bool:
    lowered = name.lower()
    return lowered in TRACKING_PARAMS or lowered.startswith(TRACKING_PREFIXES)


def _normalize_query(query: str) -> str:
    if not query:
        return ""
    pairs = [
        (key, value)
        for key, value in parse_qsl(query, keep_blank_values=True)
        if key and not _is_tracking(key)
    ]
    pairs.sort()
    return "&".join(
        f"{_percent_encode(k, _QUERY_COMPONENT_SAFE)}={_percent_encode(v, _QUERY_COMPONENT_SAFE)}"
        for k, v in pairs
    )


def normalize_url(url: str, base: str | None = None) -> str:
    """Canonical absolute URL, or `UrlNormalizationError` if it is not crawlable."""
    if not isinstance(url, str):
        raise UrlNormalizationError("URL must be a string")
    raw = _INTERNAL_WHITESPACE.sub("", url.strip())
    try:
        if base is not None:
            raw = urljoin(_INTERNAL_WHITESPACE.sub("", base.strip()), raw)
        if not raw:
            raise UrlNormalizationError("URL is empty")
        parts = urlsplit(raw)
        port = parts.port
    except ValueError as exc:
        raise UrlNormalizationError(f"malformed URL: {exc}") from exc

    scheme = parts.scheme.lower()
    if scheme not in DEFAULT_PORTS:
        raise UrlNormalizationError(f"unsupported scheme {scheme or '(none)'!r}")
    if parts.username is not None or parts.password is not None:
        raise UrlNormalizationError("URL must not contain credentials")

    host = _normalize_host(parts.hostname or "")
    netloc = host if port in (None, DEFAULT_PORTS[scheme]) else f"{host}:{port}"
    path = _remove_dot_segments(_percent_encode(parts.path, _PATH_SAFE)) or "/"
    query = _normalize_query(parts.query)
    return urlunsplit((scheme, netloc, path, query, ""))


def url_hash(normalized_url: str) -> str:
    """sha256 hex of an already-normalized URL (the `Page.url_hash` / frontier key)."""
    return hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()


def normalize_and_hash(url: str, base: str | None = None) -> tuple[str, str]:
    normalized = normalize_url(url, base)
    return normalized, url_hash(normalized)
