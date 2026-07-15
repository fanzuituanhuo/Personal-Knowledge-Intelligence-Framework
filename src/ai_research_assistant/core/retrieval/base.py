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
    ) -> None:
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
                When omitted, the implementation generates IDs internally;
                those IDs are only usable if returned by ``retrieve``.
        """
        pass


    @abstractmethod
    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant documents.

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
    ) -> None:
        """
        Delete documents from retrieval database.
        """
        pass



    @abstractmethod
    def count(self) -> int:
        """
        Return number of stored documents.
        """
        pass
