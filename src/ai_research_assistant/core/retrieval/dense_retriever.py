from typing import Any

from ai_research_assistant.database.vector_store.base import BaseVectorStore

from .base import BaseRetriever


class DenseRetriever(BaseRetriever):
    """
    Dense vector retriever backed by a vector store.

    This class delegates vector storage and similarity search to a
    ``BaseVectorStore`` implementation.
    """

    def __init__(self, vector_store: BaseVectorStore) -> None:
        self._vector_store = vector_store

    def add(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        return self._vector_store.add(
            documents=documents,
            embeddings=embeddings,
            metadata=metadata,
            ids=ids,
        )

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return self._vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

    def delete(self, ids: list[str]) -> int:
        return self._vector_store.delete(ids)

    def count(self) -> int:
        return self._vector_store.count()
