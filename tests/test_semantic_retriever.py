"""Unit tests for SemanticRetriever.

Usage:
    uv run pytest -q tests/test_semantic_retriever.py
    uv run python tests/test_semantic_retriever.py
"""

from __future__ import annotations

import pytest

from ai_research_assistant.core.embedding.base import BaseEmbedding
from ai_research_assistant.core.retrieval.config import RetrieverConfig
from ai_research_assistant.core.retrieval.factory import create_retriever
from ai_research_assistant.core.retrieval.semantic_retriever import SemanticRetriever
from ai_research_assistant.database.vector_store.faiss_store import FAISSVectorStore


DIMENSION = 4


class _FakeEmbedding(BaseEmbedding):
    """Deterministic embedding for unit tests (no real model load)."""

    DIM = DIMENSION

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vec(query)

    def _vec(self, text: str) -> list[float]:
        # Same text -> same one-hot vector within a process.
        idx = hash(text) % self.DIM
        return [1.0 if i == idx else 0.0 for i in range(self.DIM)]


@pytest.fixture
def semantic_retriever() -> SemanticRetriever:
    store = FAISSVectorStore(dimension=DIMENSION)
    retriever = create_retriever(RetrieverConfig(provider="dense"), store)
    return SemanticRetriever(
        embedding_model=_FakeEmbedding(),
        retriever=retriever,
    )


def test_retrieve_returns_relevant_chunk(
    semantic_retriever: SemanticRetriever,
) -> None:
    """Query matching an indexed document returns that chunk as top-1."""
    semantic_retriever.add(
        documents=["rate-distortion function", "orthogonal topic"],
        metadata=[{"source": "a"}, {"source": "b"}],
        ids=["rd", "other"],
    )

    results = semantic_retriever.retrieve(
        query="rate-distortion function",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["id"] == "rd"
    assert results[0]["text"] == "rate-distortion function"


def test_retrieve_top_k(semantic_retriever: SemanticRetriever) -> None:
    """top_k limits the number of returned results."""
    semantic_retriever.add(
        documents=["alpha", "beta", "gamma"],
        ids=["a", "b", "c"],
    )

    results = semantic_retriever.retrieve(query="alpha", top_k=2)

    assert len(results) == 2


def test_retrieve_with_metadata_filter(
    semantic_retriever: SemanticRetriever,
) -> None:
    """metadata_filter restricts retrieval results."""
    semantic_retriever.add(
        documents=["keep me", "drop me"],
        metadata=[{"kb": "ntc"}, {"kb": "other"}],
        ids=["keep", "drop"],
    )

    results = semantic_retriever.retrieve(
        query="keep me",
        top_k=2,
        metadata_filter={"kb": "ntc"},
    )

    assert len(results) == 1
    assert results[0]["id"] == "keep"


def test_add_embeds_and_indexes(
    semantic_retriever: SemanticRetriever,
) -> None:
    """add embeds documents and indexes them for later retrieval."""
    assert semantic_retriever.count() == 0

    ids = semantic_retriever.add(
        documents=["what is RAG?"],
        ids=["rag"],
    )

    assert ids == ["rag"]
    assert semantic_retriever.count() == 1

    results = semantic_retriever.retrieve(query="what is RAG?", top_k=1)
    assert results[0]["id"] == "rag"


def test_delete(semantic_retriever: SemanticRetriever) -> None:
    """delete removes documents from the underlying retriever."""
    semantic_retriever.add(
        documents=["a", "b"],
        ids=["a", "b"],
    )

    semantic_retriever.delete(["a"])
    assert semantic_retriever.count() == 1


def main() -> None:
    """Run the same checks as a standalone smoke test."""
    store = FAISSVectorStore(dimension=DIMENSION)
    dense = create_retriever(RetrieverConfig(), store)
    instance = SemanticRetriever(
        embedding_model=_FakeEmbedding(),
        retriever=dense,
    )

    test_retrieve_returns_relevant_chunk(instance)
    test_retrieve_top_k(
        SemanticRetriever(
            embedding_model=_FakeEmbedding(),
            retriever=create_retriever(
                RetrieverConfig(), FAISSVectorStore(dimension=DIMENSION)
            ),
        )
    )
    test_retrieve_with_metadata_filter(
        SemanticRetriever(
            embedding_model=_FakeEmbedding(),
            retriever=create_retriever(
                RetrieverConfig(), FAISSVectorStore(dimension=DIMENSION)
            ),
        )
    )
    test_add_embeds_and_indexes(
        SemanticRetriever(
            embedding_model=_FakeEmbedding(),
            retriever=create_retriever(
                RetrieverConfig(), FAISSVectorStore(dimension=DIMENSION)
            ),
        )
    )
    test_delete(
        SemanticRetriever(
            embedding_model=_FakeEmbedding(),
            retriever=create_retriever(
                RetrieverConfig(), FAISSVectorStore(dimension=DIMENSION)
            ),
        )
    )

    print("SemanticRetriever 测试全部通过。")


if __name__ == "__main__":
    main()
