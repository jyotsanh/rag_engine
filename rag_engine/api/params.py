"""Query parameters shared by list endpoints."""

from typing import Annotated

from fastapi import Query

from rag_engine.repositories import MAX_PAGE_SIZE

Limit = Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE, description="Page size.")]
Cursor = Annotated[str | None, Query(description="`next_cursor` from the previous page.")]
