from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import faiss
import numpy as np

from .base import BaseVectorStore


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-based vector store using cosine similarity.

    The implementation uses:

    - ``faiss.IndexFlatIP`` for exact inner-product search
    - ``faiss.IndexIDMap2`` for stable internal integer IDs
    - L2-normalized vectors so inner product equals cosine similarity

    Documents, metadata, and external string IDs are stored separately
    from the FAISS index.
    """

    INDEX_FILENAME = "index.faiss"
    RECORDS_FILENAME = "records.json"
    STATE_FILENAME = "state.json"

    def __init__(self, dimension: int ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")

        self._dimension = dimension

        base_index = faiss.IndexFlatIP(dimension)
        self._index = faiss.IndexIDMap2(base_index) # 真实存储的向量索引

        # internal FAISS ID -> stored record
        self._records: dict[int, dict[str, Any]] = {} #从内部整数id到存储的记录的映射，记录包含外部id、文本和元数据

        # user-facing string ID -> internal FAISS integer ID
        self._external_to_internal: dict[str, int] = {}

        self._next_internal_id = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    def add(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """
        Add documents and their embeddings to the vector store.

        Duplicate external IDs are not allowed.
        """
        self._validate_add_inputs(
            documents=documents,
            embeddings=embeddings,
            metadata=metadata,
            ids=ids,
        )

        # 如果文档列表为空，则返回空列表
        if not documents:
            return []

        record_count = len(documents)

        if metadata is None:
            metadata = [{} for _ in range(record_count)]

        if ids is None:
            ids = [self._generate_external_id() for _ in range(record_count)]

        self._validate_external_ids(ids)

        vectors = self._prepare_embeddings(embeddings)

        internal_ids = np.arange(
            self._next_internal_id,
            self._next_internal_id + record_count,
            dtype=np.int64,
        )

        # Add vectors first. Records are only changed after FAISS succeeds.
        self._index.add_with_ids(vectors, internal_ids)

        for internal_id, external_id, document, item_metadata in zip(
            internal_ids.tolist(),
            ids,
            documents,
            metadata,
            strict=True,
        ):
            self._records[internal_id] = {
                "id": external_id,
                "text": document,
                "metadata": item_metadata,
            }

            self._external_to_internal[external_id] = internal_id

        self._next_internal_id += record_count

        return ids 

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for the most similar stored documents.

        Metadata filters use exact key-value matching.
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if self.count() == 0:
            return []

        query_vector = self._prepare_query_embedding(query_embedding)

        # FAISS does not perform metadata filtering in this implementation.
        # Search all records when a filter is present, then filter the results.
        search_k = self.count() if metadata_filter else min(top_k, self.count())

        scores, internal_ids = self._index.search(query_vector, search_k)

        results: list[dict[str, Any]] = []

        for score, internal_id in zip(
            scores[0],
            internal_ids[0],
            strict=True,
        ):
            # FAISS can return -1 when fewer than k results exist.
            if internal_id == -1:
                continue

            record = self._records.get(int(internal_id))

            if record is None:
                continue

            if metadata_filter and not self._matches_metadata_filter(
                metadata=record["metadata"],
                metadata_filter=metadata_filter,
            ):
                continue

            results.append(
                {
                    "id": record["id"],
                    "text": record["text"],
                    "score": float(score),
                    "metadata": record["metadata"],
                }
            )

            if len(results) >= top_k:
                break

        return results

    def delete(self, ids: list[str]) -> int:
        """
        Delete records by their external string IDs.

        Unknown IDs are ignored.
        """
        if not ids:
            return 0

        internal_ids = [
            self._external_to_internal[external_id]
            for external_id in ids
            if external_id in self._external_to_internal
        ]

        if not internal_ids:
            return 0

        internal_ids_array = np.asarray(internal_ids, dtype=np.int64)

        removed_count = int(self._index.remove_ids(internal_ids_array))

        for internal_id in internal_ids:
            record = self._records.pop(internal_id, None)

            if record is not None:
                self._external_to_internal.pop(record["id"], None)

        return removed_count

    def count(self) -> int:
        return int(self._index.ntotal)

    def save(self, path: str | Path) -> None:
        """
        Save the FAISS index and associated records to a directory.

        Metadata must be JSON serializable.
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        index_path = directory / self.INDEX_FILENAME
        records_path = directory / self.RECORDS_FILENAME
        state_path = directory / self.STATE_FILENAME

        faiss.write_index(self._index, str(index_path))

        records_payload = [
            {
                "internal_id": internal_id,
                "id": record["id"],
                "text": record["text"],
                "metadata": record["metadata"],
            }
            for internal_id, record in self._records.items()
        ]

        with records_path.open("w", encoding="utf-8") as file:
            json.dump(
                records_payload,
                file,
                ensure_ascii=False,
                indent=2,
            )

        state_payload = {
            "dimension": self._dimension,
            "next_internal_id": self._next_internal_id,
            "record_count": self.count(),
        }

        with state_path.open("w", encoding="utf-8") as file:
            json.dump(
                state_payload,
                file,
                ensure_ascii=False,
                indent=2,
            )

    def load(self, path: str | Path) -> None:
        """
        Load a previously saved FAISS vector store.
        """
        directory = Path(path)

        index_path = directory / self.INDEX_FILENAME
        records_path = directory / self.RECORDS_FILENAME
        state_path = directory / self.STATE_FILENAME

        missing_files = [
            file_path.name
            for file_path in (index_path, records_path, state_path)
            if not file_path.exists()
        ]

        if missing_files:
            missing = ", ".join(missing_files)
            raise FileNotFoundError(
                f"Vector store files are missing from {directory}: {missing}"
            )

        loaded_index = faiss.read_index(str(index_path))

        with state_path.open("r", encoding="utf-8") as file:
            state_payload = json.load(file)

        with records_path.open("r", encoding="utf-8") as file:
            records_payload = json.load(file)

        dimension = int(state_payload["dimension"])
        next_internal_id = int(state_payload["next_internal_id"])

        if loaded_index.d != dimension:
            raise ValueError(
                "Saved state dimension does not match the FAISS index: "
                f"{dimension} != {loaded_index.d}"
            )

        records: dict[int, dict[str, Any]] = {}
        external_to_internal: dict[str, int] = {}

        for item in records_payload:
            internal_id = int(item["internal_id"])
            external_id = str(item["id"])

            if internal_id in records:
                raise ValueError(
                    f"Duplicate internal ID in saved records: {internal_id}"
                )

            if external_id in external_to_internal:
                raise ValueError(
                    f"Duplicate external ID in saved records: {external_id}"
                )

            records[internal_id] = {
                "id": external_id,
                "text": str(item["text"]),
                "metadata": item.get("metadata", {}),
            }

            external_to_internal[external_id] = internal_id

        if loaded_index.ntotal != len(records):
            raise ValueError(
                "FAISS index size does not match stored record count: "
                f"{loaded_index.ntotal} != {len(records)}"
            )

        self._index = loaded_index
        self._dimension = dimension
        self._next_internal_id = next_internal_id
        self._records = records
        self._external_to_internal = external_to_internal


    def _prepare_embeddings(
        self,
        embeddings: list[list[float]],
    ) -> np.ndarray:
        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional array")

        if vectors.shape[1] != self._dimension:
            raise ValueError(
                "Embedding dimension does not match the vector store: "
                f"{vectors.shape[1]} != {self._dimension}"
            )

        if not np.isfinite(vectors).all():
            raise ValueError("embeddings contain NaN or infinite values")

        norms = np.linalg.norm(vectors, axis=1) # 计算向量的范数

        if np.any(norms == 0):
            raise ValueError("embeddings must not contain zero vectors")

        vectors = np.ascontiguousarray(vectors) # 将向量转换为连续的内存布局
        faiss.normalize_L2(vectors) # 归一化向量

        return vectors

    def _prepare_query_embedding(
        self,
        query_embedding: list[float],
    ) -> np.ndarray:
        query_vector = np.asarray(
            [query_embedding],
            dtype=np.float32,
        )

        if query_vector.shape != (1, self._dimension):
            actual_dimension = (
                query_vector.shape[1]
                if query_vector.ndim == 2
                else "invalid"
            )

            raise ValueError(
                "Query embedding dimension does not match the vector store: "
                f"{actual_dimension} != {self._dimension}"
            )

        if not np.isfinite(query_vector).all():
            raise ValueError(
                "query_embedding contains NaN or infinite values"
            )

        if np.linalg.norm(query_vector) == 0:
            raise ValueError("query_embedding must not be a zero vector")

        query_vector = np.ascontiguousarray(query_vector)
        faiss.normalize_L2(query_vector)

        return query_vector

    # 验证添加输入的合法性
    def _validate_add_inputs(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]] | None,
        ids: list[str] | None,
    ) -> None:
        document_count = len(documents) # 文档数量

        if len(embeddings) != document_count:
            raise ValueError(
                "documents and embeddings must have the same length"
            )

        if metadata is not None and len(metadata) != document_count:
            raise ValueError(
                "documents and metadata must have the same length"
            )

        if ids is not None and len(ids) != document_count:
            raise ValueError(
                "documents and ids must have the same length"
            )

        if any(not isinstance(document, str) for document in documents):
            raise TypeError("all documents must be strings")

        if metadata is not None and any(
            not isinstance(item, dict) for item in metadata
        ):
            raise TypeError("all metadata items must be dictionaries")



    def _validate_external_ids(self, ids: list[str]) -> None:
        if any(not isinstance(external_id, str) for external_id in ids):
            raise TypeError("all IDs must be strings")

        if any(not external_id for external_id in ids):
            raise ValueError("IDs must not be empty strings")

        if len(ids) != len(set(ids)):
            raise ValueError("duplicate IDs were provided in the same batch")

        duplicate_ids = [
            external_id
            for external_id in ids
            if external_id in self._external_to_internal
        ]

        if duplicate_ids:
            raise ValueError(
                f"IDs already exist in the vector store: {duplicate_ids}"
            )


    def _generate_external_id(self) -> str:
        external_id = uuid4().hex # 随机生成一个32位十六进制字符串

        while external_id in self._external_to_internal:
            external_id = uuid4().hex # 如果生成的id已经存在，则重新生成

        return external_id

    @staticmethod
    def _matches_metadata_filter(
        metadata: dict[str, Any],
        metadata_filter: dict[str, Any],
    ) -> bool:
        return all(
            metadata.get(key) == expected_value
            for key, expected_value in metadata_filter.items()
        )
