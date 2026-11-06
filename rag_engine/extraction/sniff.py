"""What a body really is, from its bytes (servers mislabel content all the time).

- PDF: `%PDF-` within the first 1024 bytes (the spec tolerates leading junk).
- OOXML (DOCX/PPTX/XLSX): a ZIP whose entries include `[Content_Types].xml` and the
  `word/`, `ppt/` or `xl/` part folder. Only the central directory is read; nothing is
  decompressed, so a zip bomb costs nothing here.
- OLE2/CFB (`D0 CF 11 E0 …`): legacy `.doc/.xls/.ppt`, and also **encrypted** OOXML files
  (Office wraps them in an OLE container). Reported as `Sniffed.OLE`, not as a kind.
- HTML: a doctype, `<html>`, `<head>` or `<body>` near the start (after BOMs, whitespace,
  an XML prolog or comments).
"""

import io
import re
import zipfile
from enum import StrEnum

from rag_engine.domain import ContentKind

_PDF_MAGIC = b"%PDF-"
_ZIP_MAGIC = b"PK\x03\x04"
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_HEAD_BYTES = 1024
_HTML_MARKERS = re.compile(rb"<!doctype\s+html|<html[\s>]|<head[\s>]|<body[\s>]", re.I)
_TEXT_CONTROLS = frozenset(b"\t\n\r\f\x1b")  # \x1b: ANSI escapes in text dumps
_OOXML_PARTS = (
    ("word/", ContentKind.DOCX),
    ("ppt/", ContentKind.PPTX),
    ("xl/", ContentKind.XLSX),
)


class Sniffed(StrEnum):
    """Non-kind verdicts."""

    OLE = "ole"  # legacy binary Office or an encrypted OOXML file
    ZIP = "zip"  # a ZIP archive that is not an Office document


def sniff(body: bytes) -> ContentKind | Sniffed | None:
    """The content kind the bytes prove, a non-kind verdict, or None if they prove nothing."""
    head = body[:_HEAD_BYTES]
    if _PDF_MAGIC in head:
        return ContentKind.PDF
    if body.startswith(_OLE_MAGIC):
        return Sniffed.OLE
    if body.startswith(_ZIP_MAGIC):
        return _ooxml_kind(body) or Sniffed.ZIP
    if _HTML_MARKERS.search(head):
        return ContentKind.HTML
    return None


def _ooxml_kind(body: bytes) -> ContentKind | None:
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            names = archive.namelist()
    except zipfile.BadZipFile, ValueError, OSError:
        return None
    if "[Content_Types].xml" not in names:
        return None
    for prefix, kind in _OOXML_PARTS:
        if any(name.startswith(prefix) for name in names):
            return kind
    return None


def looks_like_text(body: bytes) -> bool:
    """Text in *some* encoding: no NUL bytes and almost no control characters in the head.

    Deliberately encoding-agnostic (Latin-1 or Shift-JIS pages are text too); the fetcher
    already decided the charset.
    """
    head = body[:_HEAD_BYTES]
    if not head or b"\x00" in head:
        return False
    controls = sum(1 for byte in head if byte < 0x20 and byte not in _TEXT_CONTROLS)
    return controls <= len(head) * 0.05
