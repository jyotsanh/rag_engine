from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_engine.schemas import DatabaseType


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,  # ignore empty key
        extra="ignore",  # ignores extra key if provided
    )

    # Production HOST & PORT
    PROD_HOST: str = "0.0.0.0"
    PROD_PORT: int = 8000

    # Development HOST & PORT
    DEV_HOST: str = "localhost"
    DEV_PORT: int = 8000

    # project-setting
    PROJECT_NAME: str = "RAG Engine"
    API_V1_STR: str = "/api/v1"

    # cors
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost:8000",
        "http://localhost:3000",
    ]
    ALLOW_CREDENTIALS: bool = False

    # active raw-document warehouse backend (swap without touching call sites)
    ACTIVE_DATABASE: DatabaseType = DatabaseType.MONGODB

    # database (MongoDB — raw document warehouse)
    MONGO_HOST: str = "localhost"
    MONGO_PORT: int = 27020
    MONGO_DB: str = "rag_engine"
    # Set default to None to easily detect if it's missing in .env
    MONGO_URI: str | None = None

    # ---------------------------------------------------------------------
    # Reserved for later phases — uncomment (as no-default secrets) when the
    # corresponding roadmap phase lands, so startup fails without them.
    #
    # Phase 8 — async jobs (Celery + Redis)
    # REDIS_HOST: str = "localhost"
    # REDIS_PORT: int = 6379
    #
    # Phase 9 — retrieval & generation
    # ANTHROPIC_API_KEY: SecretStr
    # VOYAGE_API_KEY: SecretStr
    # QDRANT_HOST: str = "localhost"
    # QDRANT_PORT: int = 6333
    # ---------------------------------------------------------------------

    @property
    def mongo_url(self) -> str:
        """Resolved Mongo connection string — explicit MONGO_URI wins."""
        if self.MONGO_URI:
            return self.MONGO_URI
        return f"mongodb://{self.MONGO_HOST}:{self.MONGO_PORT}"


# NOTE: once required (no-default) secrets are added in later phases, Settings()
# will look under-specified to the type checker and need a blanket ignore comment
# here. For now every field has a default, so none is needed.
settings = Settings()
