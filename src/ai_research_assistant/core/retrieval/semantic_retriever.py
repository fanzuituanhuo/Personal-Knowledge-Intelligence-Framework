from typing import Any

from ai_research_assistant.core.embedding.base import BaseEmbedding

from .base import BaseRetriever


class SemanticRetriever:
    """
    High-level semantic retriever that composes an embedding model
    with a low-level retriever.

    Converts a natural-language query into an embedding and delegates
    search to ``BaseRetriever``, hiding the embedding step from callers.
    """

    def __init__(
        self,
        embedding_model: BaseEmbedding,
        retriever: BaseRetriever,
    ) -> None:
        self._embedding_model = embedding_model
        self._retriever = retriever

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant documents for a natural-language query.

        Args:
            query:
                User question in natural language.

            top_k:
                Maximum number of results to return.

            metadata_filter:
                Optional metadata conditions used to restrict search.

        Returns:
            [
                {
                    "id": "chunk-3",
                    "text": "...",
                    "score": 0.92,
                    "metadata": {...}
                }
            ]
        """
        query_embedding = self._embedding_model.embed_query(query)
        return self._retriever.retrieve(
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

    def add(
        self,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """
        Embed documents and add them to the underlying retriever.

        Intended for reuse by a future ingestion pipeline.
        """
        embeddings = self._embedding_model.embed(documents)
        return self._retriever.add(
            documents=documents,
            embeddings=embeddings,
            metadata=metadata,
            ids=ids,
        )

    def delete(self, ids: list[str]) -> int:
        """Delete documents from the underlying retriever."""
        return self._retriever.delete(ids)

    def count(self) -> int:
        """Return the number of stored documents."""
        return self._retriever.count()
