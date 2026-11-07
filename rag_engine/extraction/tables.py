"""Markdown table rendering shared by the Office extractors (DOCX, PPTX, XLSX)."""

from collections.abc import Sequence


def cell_text(value: str) -> str:
    """One table cell: whitespace (incl. newlines) collapsed, pipes escaped."""
    return " ".join(value.split()).replace("|", "\\|")


def render_table(rows: Sequence[Sequence[str]]) -> str:
    """Rows of cell strings → a GFM table; the first non-empty row is the header.

    Rows are padded to the widest row and entirely empty rows dropped. Returns "" when
    nothing is left.
    """
    cleaned = [[cell_text(c) for c in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    padded = [row + [""] * (width - len(row)) for row in cleaned]
    lines = ["| " + " | ".join(padded[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(row) + " |" for row in padded[1:]]
    return "\n".join(lines)
