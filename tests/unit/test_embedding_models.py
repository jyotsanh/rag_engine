"""Embedding model registry: specs, dimension resolution, spaces, KB integration."""

import pytest
from pydantic import ValidationError

from rag_engine.config import EmbeddingModelName, EmbeddingProvider, settings
from rag_engine.domain import KnowledgeBase, Tenant
from rag_engine.embeddings import (
    EMBEDDING_MODELS,
    MIN_REDUCED_DIMENSIONS,
    EmbeddingSpace,
    UnsupportedEmbeddingDimensionsError,
    UnsupportedEmbeddingModelError,
    get_model_spec,
    request_dimensions,
    resolve_space,
)

SMALL = EmbeddingModelName.TEXT_EMBEDDING_3_SMALL
LARGE = EmbeddingModelName.TEXT_EMBEDDING_3_LARGE
TEN = Tenant(name="Acme").id

# ---- #
# Tests
# ---- #


class TestEmbeddingModelRegistry_S14:
    """S14 — Embedding model registry: specs, spaces, dimension validation."""

    def test_every_model_name_has_a_spec(self) -> None:
        assert set(EMBEDDING_MODELS) == set(EmbeddingModelName)
        assert all(spec.name is name for name, spec in EMBEDDING_MODELS.items())

    def test_every_spec_uses_a_known_provider(self) -> None:
        assert {s.provider for s in EMBEDDING_MODELS.values()} <= set(EmbeddingProvider)

    def test_default_model_setting_is_registered(self) -> None:
        assert settings.DEFAULT_EMBEDDING_MODEL in EMBEDDING_MODELS

    @pytest.mark.parametrize(("model", "dims"), [(SMALL, 1536), (LARGE, 3072)])
    def test_openai_specs(self, model: EmbeddingModelName, dims: int) -> None:
        spec = get_model_spec(model)

        assert spec.provider is EmbeddingProvider.OPENAI
        assert spec.native_dimensions == dims
        assert spec.supports_dimension_reduction
        assert spec.min_dimensions == MIN_REDUCED_DIMENSIONS
        assert spec.max_input_tokens == 8191
        assert spec.max_batch_inputs == 2048
        assert spec.max_batch_tokens == 300_000
        assert spec.tokenizer_encoding == "cl100k_base"

    def test_specs_are_immutable(self) -> None:
        spec = get_model_spec(SMALL)

        with pytest.raises(AttributeError):
            spec.native_dimensions = 1  # type: ignore
        with pytest.raises(TypeError):
            EMBEDDING_MODELS[SMALL] = spec  # type: ignore

    def test_lookup_accepts_plain_strings(self) -> None:
        assert get_model_spec("text-embedding-3-large").name is LARGE

    def test_unknown_model_lists_supported_ones(self) -> None:
        with pytest.raises(UnsupportedEmbeddingModelError) as exc:
            get_model_spec("text-embedding-ada-002")

        assert exc.value.status_code == 422
        assert set(exc.value.details["supported"]) == {m.value for m in EmbeddingModelName}

    def test_resolve_defaults_to_native_dimensions(self) -> None:
        assert resolve_space(LARGE) == EmbeddingSpace(LARGE, 3072)

    @pytest.mark.parametrize(
        ("model", "dims"), [(LARGE, 1024), (LARGE, 256), (SMALL, 512), (SMALL, 1536)]
    )
    def test_resolve_accepts_reduced_dimensions(self, model: EmbeddingModelName, dims: int) -> None:
        assert resolve_space(model, dims).dimensions == dims

    @pytest.mark.parametrize(
        ("model", "dims"), [(SMALL, 3072), (SMALL, 255), (LARGE, 0), (LARGE, -1), (LARGE, 4096)]
    )
    def test_resolve_rejects_out_of_range(self, model: EmbeddingModelName, dims: int) -> None:
        with pytest.raises(UnsupportedEmbeddingDimensionsError) as exc:
            resolve_space(model, dims)

        assert exc.value.details["requested"] == dims
        assert exc.value.details["max"] == get_model_spec(model).native_dimensions

    def test_space_key_is_stable_and_collection_safe(self) -> None:
        key = resolve_space(LARGE, 1024).key

        assert key == "text-embedding-3-large__1024"
        assert all(ch.isalnum() or ch in "-_" for ch in key)

    def test_spaces_differ_by_model_and_dimensions(self) -> None:
        keys = {
            resolve_space(SMALL).key,
            resolve_space(LARGE).key,
            resolve_space(LARGE, 1536).key,
        }

        assert len(keys) == 3  # same dims, different model: still not comparable

    def test_request_dimensions_only_sent_when_reduced(self) -> None:
        assert request_dimensions(resolve_space(SMALL)) is None
        assert request_dimensions(resolve_space(LARGE, 1024)) == 1024

    def test_knowledge_base_validates_dimensions_for_its_model(self) -> None:
        kb = KnowledgeBase(
            tenant_id=TEN, name="docs", embedding_model=LARGE, embedding_dimensions=1024
        )

        assert kb.embedding_space == EmbeddingSpace(LARGE, 1024)
        with pytest.raises(ValidationError, match="not supported"):
            KnowledgeBase(
                tenant_id=TEN, name="docs", embedding_model=SMALL, embedding_dimensions=3072
            )
