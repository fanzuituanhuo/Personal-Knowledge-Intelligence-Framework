from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseVectorStore(ABC):
    """
    Abstract base class for vector storage backends.

    A vector store is responsible for:

    - storing vectors and their associated documents
    - performing vector similarity search
    - deleting stored records
    - persisting and restoring the vector index

    It does not generate embeddings or decide retrieval strategies.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Return the dimension of vectors stored in this vector store.
        """
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
        Add vectors and their associated documents to the store.

        Args:
            documents:
                Text chunks to be added.

            embeddings:
                Vector representations of the documents.

            metadata:
                Additional information per document, same length as
                ``documents`` when provided:
                its length must match ``documents``.

                Example:
                    {
                        "source": "paper.pdf",
                        "page": 10,
                        "chunk_id": 3,
                    }

            ids:
                Optional stable identifiers for each document. When
                provided, its length must match ``documents``.

                When omitted, the vector store generates identifiers.

        Returns:
            IDs of the added records, in the same order as the input
            documents.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for vectors most similar to the query vector.

        Args:
            query_embedding:
                Vector representation of the query.

            top_k:
                Maximum number of results to return.

            metadata_filter:
                Optional metadata conditions used to restrict search.

                Example:
                    {
                        "knowledge_base": "ntc",
                        "source": "paper.pdf",
                    }

                Backends that do not support native filtering may apply
                the filter after vector search.

        Returns:
            Search results ordered from most relevant to least relevant.

            Example:
                [
                    {
                        "id": "chunk-3",
                        "text": "...",
                        "score": 0.92,
                        "metadata": {
                            "source": "paper.pdf",
                            "page": 10,
                        },
                    }
                ]

        Notes:
            Higher scores should represent greater relevance. Concrete
            implementations should convert their native distance values
            into this convention when necessary.
        """
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> int:
        """
        Delete records by their IDs.

        Args:
            ids:
                IDs of records to delete.

        Returns:
            Number of records successfully deleted.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Return the total number of records in the vector store.
        """
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """
        Persist the vector index and associated data to disk.

        Args:
            path:
                Directory used to store the vector index, documents,
                metadata, IDs, and other required state.
        """
        pass

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """
        Restore the vector index and associated data from disk.

        Args:
            path:
                Directory containing a previously saved vector store.
        """
        pass
