"""Base document abstractions (Phase 2).

Home of ``NoSQLBaseDocument`` (MongoDB ODM). ``VectorBaseDocument`` (Qdrant
vector records) arrives in a later phase.
"""

from .nosql import NoSQLBaseDocument

__all__ = ["NoSQLBaseDocument"]
