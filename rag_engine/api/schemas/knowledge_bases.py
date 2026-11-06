from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_engine.domain import EmbeddingModelName, KnowledgeBase, ResourceStatus
from rag_engine.domain.types import Description, Name


class KnowledgeBaseCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Name
    description: Description = ""
    embedding_model: EmbeddingModelName | None = Field(
        default=None,
        description="Defaults to the tenant's default model. Cannot be changed later.",
    )
    embedding_dimensions: int | None = Field(
        default=None,
        gt=0,
        description="Shortened vector size (≥256, ≤ the model's native size). "
        "Defaults to the native size. Cannot be changed later.",
    )


class KnowledgeBasePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Name | None = None
    description: Description | None = None

    @model_validator(mode="after")
    def _meaningful(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("provide at least one field to change")
        nulls = sorted(f for f in self.model_fields_set if getattr(self, f) is None)
        if nulls:
            raise ValueError(f"fields cannot be null: {nulls}")
        return self


class KnowledgeBaseResponse(BaseModel):
    id: str
    name: str
    description: str
    status: ResourceStatus
    embedding_model: EmbeddingModelName
    embedding_dimensions: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, kb: KnowledgeBase) -> KnowledgeBaseResponse:
        return cls.model_validate(kb.model_dump(exclude={"tenant_id"}))
