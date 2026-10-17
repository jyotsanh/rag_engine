"""Opaque keyset pagination over `_id` (UUIDv7 ids are time-ordered)."""

import base64
import binascii
import json
from typing import Literal

from pydantic import BaseModel, Field

from rag_engine.repositories.exceptions import InvalidCursorError

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200

SortOrder = Literal["asc", "desc"]


class Page[T](BaseModel):
    items: list[T]
    next_cursor: str | None = Field(
        default=None, description="Pass back as `cursor` to fetch the next page; null at the end."
    )


def encode_cursor(last_id: str, order: SortOrder) -> str:
    raw = json.dumps({"after": last_id, "order": order}, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str, order: SortOrder) -> str:
    """Return the `_id` to continue after; the cursor must come from a same-order listing."""
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded.encode()))
        after, cursor_order = data["after"], data["order"]
    except (binascii.Error, ValueError, KeyError, TypeError) as exc:
        raise InvalidCursorError() from exc
    if not isinstance(after, str) or cursor_order != order:
        raise InvalidCursorError()
    return after
