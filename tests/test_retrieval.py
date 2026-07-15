"""Unit tests for DenseRetriever and its factory.

Usage:
    uv run pytest -q tests/test_retrieval.py
    uv run python tests/test_retrieval.py
"""

from __future__ import annotations

import pytest

from ai_research_assistant.core.retrieval.config import RetrieverConfig
from ai_research_assistant.core.retrieval.dense_retriever import DenseRetriever
from ai_research_assistant.core.retrieval.factory import create_retriever
from ai_research_assistant.database.vector_store.faiss_store import FAISSVectorStore


DIMENSION = 4


@pytest.fixture
def vector_store() -> FAISSVectorStore:
    return FAISSVectorStore(dimension=DIMENSION)


@pytest.fixture
def retriever(vector_store: FAISSVectorStore) -> DenseRetriever:
    return create_retriever(RetrieverConfig(provider="dense"), vector_store)


def test_factory_creates_dense_retriever(vector_store: FAISSVectorStore) -> None:
    """create_retriever returns a DenseRetriever for provider='dense'."""
    instance = create_retriever(RetrieverConfig(), vector_store)
    assert isinstance(instance, DenseRetriever)


def test_factory_rejects_unknown_provider(vector_store: FAISSVectorStore) -> None:
    """Unsupported providers raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported retriever provider"):
        create_retriever(RetrieverConfig(provider="unknown"), vector_store)


def test_add_and_retrieve(retriever: DenseRetriever) -> None:
    """Added documents can be retrieved by nearest embedding."""
    ids = retriever.add(
        documents=["exact match", "orthogonal"],
        embeddings=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        metadata=[{"source": "a"}, {"source": "b"}],
        ids=["exact", "orthogonal"],
    )

    assert ids == ["exact", "orthogonal"]
    assert retriever.count() == 2

    results = retriever.retrieve(
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["id"] == "exact"
    assert results[0]["text"] == "exact match"


def test_retrieve_with_metadata_filter(retriever: DenseRetriever) -> None:
    """metadata_filter restricts retrieval results."""
    retriever.add(
        documents=["keep", "drop"],
        embeddings=[[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]],
        metadata=[{"kb": "ntc"}, {"kb": "other"}],
        ids=["keep", "drop"],
    )

    results = retriever.retrieve(
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        top_k=2,
        metadata_filter={"kb": "ntc"},
    )

    assert len(results) == 1
    assert results[0]["id"] == "keep"


def test_delete(retriever: DenseRetriever) -> None:
    """delete removes documents from the underlying store."""
    retriever.add(
        documents=["a", "b"],
        embeddings=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        ids=["a", "b"],
    )

    deleted = retriever.delete(["a"])
    assert deleted == 1
    assert retriever.count() == 1


def main() -> None:
    """Run the same checks as a standalone smoke test."""
    store = FAISSVectorStore(dimension=DIMENSION)
    instance = create_retriever(RetrieverConfig(), store)

    test_factory_creates_dense_retriever(store)
    test_factory_rejects_unknown_provider(store)
    test_add_and_retrieve(instance)
    test_retrieve_with_metadata_filter(
        create_retriever(RetrieverConfig(), FAISSVectorStore(dimension=DIMENSION))
    )
    test_delete(create_retriever(RetrieverConfig(), FAISSVectorStore(dimension=DIMENSION)))

    print("Retrieval 测试全部通过。")


if __name__ == "__main__":
    main()
