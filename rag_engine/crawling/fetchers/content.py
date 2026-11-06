"""Content-type allowlist and HTML charset detection."""

import codecs
import re
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from charset_normalizer import from_bytes

from rag_engine.domain import ContentKind

MIME_KINDS: dict[str, ContentKind] = {
    "text/html": ContentKind.HTML,
    "application/xhtml+xml": ContentKind.HTML,
    "application/pdf": ContentKind.PDF,
    "application/x-pdf": ContentKind.PDF,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ContentKind.DOCX,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ContentKind.PPTX,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ContentKind.XLSX,
}
EXTENSION_KINDS: dict[str, ContentKind] = {
    ".html": ContentKind.HTML,
    ".htm": ContentKind.HTML,
    ".xhtml": ContentKind.HTML,
    ".pdf": ContentKind.PDF,
    ".docx": ContentKind.DOCX,
    ".pptx": ContentKind.PPTX,
    ".xlsx": ContentKind.XLSX,
}
# servers that mislabel downloads: trust the URL extension instead
GENERIC_MIME_TYPES = frozenset({"application/octet-stream", "binary/octet-stream", ""})

ACCEPT_HEADER = ", ".join(
    [
        "text/html",
        "application/xhtml+xml;q=0.9",
        "application/pdf;q=0.8",
        *(m + ";q=0.8" for m in MIME_KINDS if "officedocument" in m),
        "*/*;q=0.1",
    ]
)

_META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9._:-]+)""", re.I)
_BOMS = (
    (codecs.BOM_UTF8, "utf-8"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)


def parse_content_type(header: str | None) -> tuple[str, str | None]:
    """`text/html; charset=UTF-8` → (`text/html`, `utf-8`)."""
    if not header:
        return "", None
    mime, _, params = header.partition(";")
    charset = None
    for param in params.split(";"):
        key, _, value = param.strip().partition("=")
        if key.lower() == "charset" and value:
            charset = value.strip().strip("\"'").lower()
    return mime.strip().lower(), charset


def kind_for(mime: str, url: str) -> ContentKind | None:
    """Allowed kind for a response, or `None` if the content must be skipped."""
    if mime in MIME_KINDS:
        return MIME_KINDS[mime]
    if mime in GENERIC_MIME_TYPES:
        return EXTENSION_KINDS.get(PurePosixPath(urlsplit(url).path).suffix.lower())
    return None


def _usable(encoding: str | None) -> str | None:
    if not encoding:
        return None
    try:
        return codecs.lookup(encoding).name
    except LookupError:
        return None


def decode_html(body: bytes, header_charset: str | None) -> tuple[str, str]:
    """Decode HTML bytes → (text, encoding used).

    Order: HTTP header charset → BOM → `<meta charset>` / `http-equiv` in the first
    4 KB → statistical detection → UTF-8 with replacement characters.
    """
    declared = _usable(header_charset)
    for bom, encoding in _BOMS:
        # a BOM beats a missing or Unicode header charset (WHATWG "BOM sniffing")
        if body.startswith(bom) and (declared is None or declared.startswith("utf")):
            return body[len(bom) :].decode(encoding, errors="replace"), encoding
    if declared is not None:
        return body.decode(declared, errors="replace"), declared
    match = _META_CHARSET.search(body[:4096])
    if match and (meta := _usable(match.group(1).decode("ascii", "ignore"))) is not None:
        return body.decode(meta, errors="replace"), meta
    best = from_bytes(body[:65536]).best()
    if best is not None and (detected := _usable(best.encoding)) is not None:
        return body.decode(detected, errors="replace"), detected
    return body.decode("utf-8", errors="replace"), "utf-8"
