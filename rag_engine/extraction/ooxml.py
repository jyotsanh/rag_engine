"""Shared guard for Office Open XML (DOCX/PPTX/XLSX) archives.

They are ZIP files, and their parsers inflate parts into memory. A crafted file can
declare a 2 GB `document.xml` that compresses to kilobytes (a zip bomb), so the total
declared uncompressed size is checked against `OOXML_MAX_UNCOMPRESSED_BYTES` from the
central directory alone, before any parser touches the data.
"""

import io
import zipfile

from rag_engine.extraction.exceptions import DocumentTooLargeError, ExtractionError


def check_archive(body: bytes, max_uncompressed: int) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            declared = sum(info.file_size for info in archive.infolist())
    except (zipfile.BadZipFile, ValueError, OSError) as exc:
        raise ExtractionError(f"not a readable Office archive: {exc}", detail="corrupt") from exc
    if declared > max_uncompressed:
        raise DocumentTooLargeError(
            f"archive inflates to {declared} bytes, over {max_uncompressed}",
            detail="uncompressed",
        )
