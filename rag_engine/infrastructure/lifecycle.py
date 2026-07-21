from fastapi import FastAPI
from loguru import logger

from rag_engine.config import settings
from rag_engine.infrastructure.db import BaseDatabase, DatabaseDispatcher


class StartupApplication:
    """Bootstraps and tears down the application's backing resources.

    ``start`` resolves + connects every resource the app needs and stashes it on
    ``app.state``; ``stop`` releases them. Both are classmethods so the lifespan
    reads as ``StartupApplication.start(app)`` / ``StartupApplication.stop(app)``.

    Resources are added one ``_init_*`` method at a time as later phases land
    (Redis + Celery in Phase 8, Qdrant in Phase 9).
    """

    @classmethod
    async def start(cls, app: FastAPI) -> None:
        # a registry mirrors the house pattern: lazily cache resolved backends by type
        app.state.database_registry = {}

        await cls._init_database(app)

    @classmethod
    async def stop(cls, app: FastAPI) -> None:
        database: BaseDatabase | None = getattr(app.state, "database", None)
        if database is not None:
            logger.info("closing database connection...")
            database.close()
            logger.info("database connection closed.")

    @classmethod
    async def _init_database(cls, app: FastAPI) -> None:
        """Resolve the active DB backend via the dispatcher, connect, and verify."""
        try:
            database = DatabaseDispatcher.dispatch(settings.ACTIVE_DATABASE)
            database.connect()
            database.ping()  # fail fast if the backend is unreachable

            app.state.database_registry[settings.ACTIVE_DATABASE] = database
            app.state.database = database
            logger.info(f"active database `{settings.ACTIVE_DATABASE}` initialized")
        except Exception as db_error:
            logger.error(f"database `{settings.ACTIVE_DATABASE}` init failed: {db_error}")
            raise RuntimeError(f"Database unavailable at startup: {db_error}") from db_error
