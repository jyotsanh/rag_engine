"""loguru is the single logging entry point for the whole app.

Import the configured logger anywhere with ``from rag_engine.config import logger``;
never use the stdlib ``logging`` module directly.
"""

from loguru import logger

__all__ = ["logger"]
