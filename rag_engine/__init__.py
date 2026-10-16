"""rag_engine: website → chunks → embeddings → vector store, with a search API.

``uv run dev`` / ``uv run server`` start the API through the functions below.
"""

import uvicorn
from loguru import logger

APP_PATH = "rag_engine.api.app:app"

# Step 3 replaces these with values from `rag_engine.config.settings`
DEV_HOST, DEV_PORT = "127.0.0.1", 8000
PROD_HOST, PROD_PORT = "0.0.0.0", 8000


def dev() -> None:
    """
    running server in development mode (auto-reload)
    """
    logger.info("running server **development** mode")
    uvicorn.run(APP_PATH, host=DEV_HOST, port=DEV_PORT, reload=True)


def server() -> None:
    """
    running server in production mode
    """
    logger.info("running server **production** mode")
    uvicorn.run(APP_PATH, host=PROD_HOST, port=PROD_PORT)


__all__ = ["APP_PATH", "dev", "server"]
