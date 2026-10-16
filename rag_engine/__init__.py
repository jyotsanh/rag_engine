"""rag_engine: website → chunks → embeddings → vector store, with a search API.

``uv run dev`` / ``uv run server`` start the API through the functions below.
"""

import uvicorn

from rag_engine.config import logger, settings, setup_logging

APP_PATH = "rag_engine.api.app:app"
__version__ = "0.1.0"


def dev() -> None:
    """
    running server in development mode (auto-reload)
    """
    setup_logging()
    logger.info("running server **development** mode")
    # log_config=None: uvicorn must not install its own handlers over the loguru intercept
    uvicorn.run(
        APP_PATH,
        host=settings.DEV_HOST,
        port=settings.DEV_PORT,
        reload=True,
        log_config=None,
        access_log=False,  # ProcessTimeMiddleware writes the access log with request_id
    )


def server() -> None:
    """
    running server in production mode
    """
    setup_logging()
    logger.info("running server **production** mode")
    uvicorn.run(
        APP_PATH,
        host=settings.PROD_HOST,
        port=settings.PROD_PORT,
        log_config=None,
        access_log=False,
    )


__all__ = ["APP_PATH", "__version__", "dev", "server"]
