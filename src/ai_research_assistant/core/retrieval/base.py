from abc import ABC, abstractmethod
from typing import Any


class BaseRetriever(ABC):
    """
    Abstract base class for retrieval systems.

    Responsible for:
    - storing document embeddings
    - searching relevant documents
    - managing retrieval index
    """

    def __init__(self):
        pass

    @abstractmethod
    def add(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """
        Add documents and their embeddings into the index.

        Args:
            documents:
                Text chunks.

            embeddings:
                Vector representations of documents.

            metadata:
                Additional information per document, same length as
                ``documents`` when provided:
                {
                    "source": "paper.pdf",
                    "page": 10,
                    "chunk_id": 3
                }

            ids:
                Stable document identifiers, same length as ``documents``
                when provided. Used for upsert and later ``delete``.
                When omitted, the implementation generates IDs internally.

        Returns:
            IDs of the added documents, in the same order as the input.
        """
        pass


    @abstractmethod
    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant documents.

        Args:
            query_embedding:
                Vector representation of the query.

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
        pass


    @abstractmethod
    def delete(
        self,
        ids: list[str],
    ) -> int:
        """
        Delete documents from retrieval database.

        Returns:
            Number of documents successfully deleted.
        """
        pass



    @abstractmethod
    def count(self) -> int:
        """
        Return number of stored documents.
        """
        pass
