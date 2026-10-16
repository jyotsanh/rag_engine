"""rag_engine: website → chunks → embeddings → vector store, with a search API.

``uv run dev`` / ``uv run server`` start the API through the functions below.
"""

import uvicorn
from loguru import logger

from rag_engine.config import settings

APP_PATH = "rag_engine.api.app:app"


def dev() -> None:
    """
    running server in development mode (auto-reload)
    """
    logger.info("running server **development** mode")
    uvicorn.run(APP_PATH, host=settings.DEV_HOST, port=settings.DEV_PORT, reload=True)


def server() -> None:
    """
    running server in production mode
    """
    logger.info("running server **production** mode")
    uvicorn.run(APP_PATH, host=settings.PROD_HOST, port=settings.PROD_PORT)


__all__ = ["APP_PATH", "dev", "server"]
