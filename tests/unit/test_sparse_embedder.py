"""Sparse BM25 embedder (fastembed `Qdrant/bm25`): vectors, vocabulary, threading, loading."""

import asyncio
import threading
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from rag_engine.config import Settings, settings
from rag_engine.embeddings import (
    EmbeddingInputError,
    EmbeddingResponseError,
    FastembedBm25Embedder,
    SparseVector,
    download_model,
    get_sparse_embedder,
)


@pytest.fixture(scope="module")
def bm25() -> FastembedBm25Embedder:
    """The real model, loaded once (downloaded into the fastembed cache on first use)."""
    embedder = get_sparse_embedder()
    assert isinstance(embedder, FastembedBm25Embedder)
    return embedder


def _weights(vector: SparseVector) -> dict[int, float]:
    return dict(zip(vector.indices, vector.values, strict=True))


class _FakeModel:
    """A fastembed-like model returning one fixed vector per text; records the threads."""

    def __init__(self, *, drop_one: bool = False) -> None:
        self.threads: list[str] = []
        self.drop_one = drop_one

    def _vector(self) -> Any:
        return SimpleNamespace(indices=np.array([7, 3, 7]), values=np.array([0.5, 1.0, 0.25]))

    def embed(self, texts: list[str]) -> Iterator[Any]:
        self.threads.append(threading.current_thread().name)
        count = len(texts) - (1 if self.drop_one else 0)
        return iter([self._vector() for _ in range(count)])

    def query_embed(self, text: str) -> Iterator[Any]:
        self.threads.append(threading.current_thread().name)
        return iter([self._vector()])


# ---- #
# Tests
# ---- #


class TestSparseEmbedder_S66:
    """S66 — Sparse embedder: BM25 documents and queries, validation, lazy shared model."""

    # ---- the real BM25 model ---- #

    async def test_documents_get_well_formed_bm25_vectors(
        self, bm25: FastembedBm25Embedder
    ) -> None:
        texts = ["The WX-2048 sorts 1200 widgets per minute.", "Calibrate the torque sensor."]

        vectors = await bm25.embed_documents(texts)

        assert len(vectors) == 2
        for vector in vectors:
            assert len(vector) > 0
            assert list(vector.indices) == sorted(set(vector.indices))
            assert all(0 <= i < 2**32 for i in vector.indices)
            assert all(v > 0 for v in vector.values)
        assert await bm25.embed_documents(texts) == vectors  # deterministic

    async def test_queries_share_the_document_vocabulary(self, bm25: FastembedBm25Embedder) -> None:
        (document,) = await bm25.embed_documents(["The WX-2048 sorter is calibrated weekly."])

        query = await bm25.embed_query("WX-2048 calibration")

        assert set(query.indices) <= set(document.indices)  # stemmed: calibration ≈ calibrated
        assert set(query.values) == {1.0}  # IDF is Qdrant's job (modifier), not ours

    async def test_stemming_and_stopwords(self, bm25: FastembedBm25Embedder) -> None:
        sorting, sorts = await bm25.embed_documents(["sorting", "sorts"])
        (stopwords,) = await bm25.embed_documents(["the and of to"])

        assert sorting.indices == sorts.indices  # one stem
        assert len(stopwords) == 0  # a valid, empty vector

    async def test_term_weight_saturates_with_document_length(
        self, bm25: FastembedBm25Embedder
    ) -> None:
        filler = " ".join(f"word{i}" for i in range(300))
        short, long_ = await bm25.embed_documents(["torque sensor", f"torque sensor {filler}"])
        (term,) = (await bm25.embed_query("torque")).indices

        assert _weights(short)[term] > _weights(long_)[term]

    # ---- the contract (scripted model) ---- #

    async def test_an_empty_batch_never_loads_the_model(self) -> None:
        loads = 0

        def loader() -> _FakeModel:
            nonlocal loads
            loads += 1
            return _FakeModel()

        embedder = FastembedBm25Embedder(loader=loader)

        assert await embedder.embed_documents([]) == []
        assert loads == 0

    @pytest.mark.parametrize("bad", ["", "  \n"])
    async def test_empty_texts_are_refused(self, bad: str) -> None:
        embedder = FastembedBm25Embedder(loader=_FakeModel)

        with pytest.raises(EmbeddingInputError) as exc:
            await embedder.embed_documents(["fine", bad])
        with pytest.raises(EmbeddingInputError):
            await embedder.embed_query(bad)

        assert exc.value.details == {"index": 1}

    async def test_the_model_runs_off_the_event_loop(self) -> None:
        model = _FakeModel()
        embedder = FastembedBm25Embedder(loader=lambda: model)

        (vector,) = await embedder.embed_documents(["a"])
        await embedder.embed_query("q")

        assert model.threads and threading.main_thread().name not in model.threads
        assert vector == SparseVector((3, 7), (1.0, 0.75))  # sorted, duplicates summed

    async def test_a_model_returning_the_wrong_count_is_rejected(self) -> None:
        embedder = FastembedBm25Embedder(loader=lambda: _FakeModel(drop_one=True))

        with pytest.raises(EmbeddingResponseError, match="expected 2 sparse vectors, got 1"):
            await embedder.embed_documents(["a", "b"])

    async def test_concurrent_first_use_loads_the_model_once(self) -> None:
        loads = 0

        def loader() -> _FakeModel:
            nonlocal loads
            loads += 1
            threading.Event().wait(0.05)  # a slow load, so the threads race
            return _FakeModel()

        embedder = FastembedBm25Embedder(loader=loader)

        await asyncio.gather(*(embedder.embed_documents([f"t{i}"]) for i in range(8)))

        assert loads == 1

    # ---- SparseVector ---- #

    def test_sparse_vector_validation(self) -> None:
        assert len(SparseVector((), ())) == 0
        with pytest.raises(ValueError, match="same length"):
            SparseVector((1, 2), (1.0,))
        with pytest.raises(ValueError, match="sorted and unique"):
            SparseVector((2, 1), (1.0, 1.0))
        with pytest.raises(ValueError, match="sorted and unique"):
            SparseVector((1, 1), (1.0, 1.0))
        with pytest.raises(ValueError, match="uint32"):
            SparseVector((2**32,), (1.0,))
        with pytest.raises(ValueError, match="finite and positive"):
            SparseVector((1,), (0.0,))
        with pytest.raises(ValueError, match="finite and positive"):
            SparseVector((1,), (float("nan"),))

    def test_from_pairs_sorts_merges_and_drops_zeros(self) -> None:
        vector = SparseVector.from_pairs([9, 2, 9, 5], [0.5, 1.0, 0.25, 0.0])

        assert vector == SparseVector((2, 9), (1.0, 0.75))

    # ---- dispatch and image baking ---- #

    def test_the_process_shares_one_embedder(self, override_settings) -> None:
        assert get_sparse_embedder() is get_sparse_embedder()
        other = Settings(_env_file=None, SPARSE_EMBEDDING_LANGUAGE="german")
        assert get_sparse_embedder(config=other) is not get_sparse_embedder()
        assert get_sparse_embedder(config=other).config is other

    def test_download_model_caches_the_files(self, bm25: FastembedBm25Embedder) -> None:
        assert download_model() == settings.SPARSE_EMBEDDING_MODEL == "Qdrant/bm25"
        assert bm25.model() is not None
