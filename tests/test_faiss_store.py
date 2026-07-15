"""Unit tests for FAISSVectorStore.

Usage:
    uv run pytest -q tests/test_faiss_store.py
    uv run python tests/test_faiss_store.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_research_assistant.database.vector_store.faiss_store import FAISSVectorStore


DIMENSION = 4


@pytest.fixture
def store() -> FAISSVectorStore:
    return FAISSVectorStore(dimension=DIMENSION)


def test_add_updates_count(store: FAISSVectorStore) -> None:
    """Adding documents increases the store count by the batch size."""
    assert store.count() == 0

    store.add(
        documents=["doc-a", "doc-b"],
        embeddings=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        ids=["a", "b"],
    )

    assert store.count() == 2


def test_similar_vector_ranks_first(store: FAISSVectorStore) -> None:
    """The nearest stored vector should appear first in search results."""
    store.add(
        documents=["exact match", "orthogonal", "nearby"],
        embeddings=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
        ],
        ids=["exact", "orthogonal", "nearby"],
    )

    results = store.search(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=3)

    assert len(results) == 3
    assert results[0]["id"] == "exact"
    assert results[0]["text"] == "exact match"
    assert results[0]["score"] >= results[1]["score"] >= results[2]["score"]


def test_auto_generates_ids(store: FAISSVectorStore) -> None:
    """When ids are omitted, unique non-empty string IDs are generated."""
    ids = store.add(
        documents=["auto-1", "auto-2"],
        embeddings=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )

    assert len(ids) == 2
    assert all(isinstance(item, str) and item for item in ids)
    assert len(set(ids)) == 2
    assert store.count() == 2


def test_duplicate_id_raises(store: FAISSVectorStore) -> None:
    """Reusing an existing external ID raises ValueError."""
    store.add(
        documents=["first"],
        embeddings=[[1.0, 0.0, 0.0, 0.0]],
        ids=["dup"],
    )

    with pytest.raises(ValueError, match="already exist"):
        store.add(
            documents=["second"],
            embeddings=[[0.0, 1.0, 0.0, 0.0]],
            ids=["dup"],
        )


def test_metadata_filter(store: FAISSVectorStore) -> None:
    """Metadata filters restrict search results to exact key-value matches."""
    store.add(
        documents=["ntc paper", "other paper", "ntc note"],
        embeddings=[
            [1.0, 0.0, 0.0, 0.0],
            [0.99, 0.01, 0.0, 0.0],
            [0.98, 0.02, 0.0, 0.0],
        ],
        metadata=[
            {"knowledge_base": "ntc", "source": "paper.pdf"},
            {"knowledge_base": "other", "source": "paper.pdf"},
            {"knowledge_base": "ntc", "source": "note.md"},
        ],
        ids=["ntc-paper", "other-paper", "ntc-note"],
    )

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        top_k=5,
        metadata_filter={"knowledge_base": "ntc"},
    )

    assert {item["id"] for item in results} == {"ntc-paper", "ntc-note"}
    assert all(item["metadata"]["knowledge_base"] == "ntc" for item in results)


def test_delete(store: FAISSVectorStore) -> None:
    """Deleting by ID removes records and updates the count."""
    store.add(
        documents=["keep", "remove"],
        embeddings=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        ids=["keep", "remove"],
    )

    removed = store.delete(["remove", "missing"])

    assert removed == 1
    assert store.count() == 1

    results = store.search(query_embedding=[0.0, 1.0, 0.0, 0.0], top_k=5)
    assert [item["id"] for item in results] == ["keep"]


def test_save_load_roundtrip(store: FAISSVectorStore, tmp_path: Path) -> None:
    """Search results remain consistent after save and load."""
    store.add(
        documents=["alpha", "beta"],
        embeddings=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        metadata=[{"tag": "a"}, {"tag": "b"}],
        ids=["alpha", "beta"],
    )

    before = store.search(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=2)

    save_dir = tmp_path / "faiss_store"
    store.save(save_dir)

    restored = FAISSVectorStore(dimension=DIMENSION)
    restored.load(save_dir)

    after = restored.search(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=2)

    assert restored.count() == store.count()
    assert restored.dimension == store.dimension
    assert after == before


def test_wrong_embedding_dimension_raises(store: FAISSVectorStore) -> None:
    """Adding or searching with the wrong vector dimension raises ValueError."""
    with pytest.raises(ValueError, match="dimension"):
        store.add(
            documents=["bad"],
            embeddings=[[1.0, 0.0, 0.0]],
            ids=["bad"],
        )

    store.add(
        documents=["ok"],
        embeddings=[[1.0, 0.0, 0.0, 0.0]],
        ids=["ok"],
    )

    with pytest.raises(ValueError, match="dimension"):
        store.search(query_embedding=[1.0, 0.0, 0.0], top_k=1)


def main() -> None:
    """Run the same checks as a standalone smoke test."""
    import tempfile

    instance = FAISSVectorStore(dimension=DIMENSION)

    test_add_updates_count(instance)

    instance = FAISSVectorStore(dimension=DIMENSION)
    test_similar_vector_ranks_first(instance)

    instance = FAISSVectorStore(dimension=DIMENSION)
    test_auto_generates_ids(instance)

    instance = FAISSVectorStore(dimension=DIMENSION)
    test_duplicate_id_raises(instance)

    instance = FAISSVectorStore(dimension=DIMENSION)
    test_metadata_filter(instance)

    instance = FAISSVectorStore(dimension=DIMENSION)
    test_delete(instance)

    instance = FAISSVectorStore(dimension=DIMENSION)
    with tempfile.TemporaryDirectory() as tmp:
        test_save_load_roundtrip(instance, Path(tmp))

    instance = FAISSVectorStore(dimension=DIMENSION)
    test_wrong_embedding_dimension_raises(instance)

    print("FAISSVectorStore 测试全部通过。")


if __name__ == "__main__":
    main()
