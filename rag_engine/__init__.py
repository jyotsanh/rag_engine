import uvicorn
from loguru import logger

from rag_engine.config import settings


def dev():
    """
    running server in development mode (auto-reload)
    """
    logger.info("running server **development** mode")
    uvicorn.run(
        "rag_engine.api.app:app",
        host=settings.DEV_HOST,
        port=settings.DEV_PORT,
        reload=True,
    )


def server():
    """
    running server in production mode
    """
    logger.info("running server **production** mode")
    uvicorn.run(
        "rag_engine.api.app:app",
        host=settings.PROD_HOST,
        port=settings.PROD_PORT,
    )


if __name__ == "__main__":
    dev()
