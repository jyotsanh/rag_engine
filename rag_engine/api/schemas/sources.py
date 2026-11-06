from datetime import datetime

from pydantic import BaseModel, Field

from rag_engine.domain import ResourceStatus, Source, SourceStats


class SourceResponse(BaseModel):
    id: str
    kb_id: str
    seed_url: str
    status: ResourceStatus
    active_version: str | None = Field(
        description="Crawl job whose content search currently serves; null until a crawl succeeds."
    )
    last_job_id: str | None = Field(description="Most recent crawl job for this source.")
    stats: SourceStats
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, source: Source) -> SourceResponse:
        return cls.model_validate(source.model_dump(exclude={"tenant_id"}))
