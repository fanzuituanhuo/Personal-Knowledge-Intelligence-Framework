from typing import List
from sentence_transformers import SentenceTransformer

from .base import BaseEmbedding
from .config import EmbeddingConfig


class BGEEmbedding(BaseEmbedding):
    """
    BGE-M3 embedding implementation
    """

    def __init__(
        self,
        model_name: str = EmbeddingConfig.model_name,
        device: str = EmbeddingConfig.device,
    ):
        """
        Args:
            model_name:
                HuggingFace模型名称或本地路径

            device:
                cpu / cuda / mps
        """

        self.model_name = model_name
        self.device = device
        # 使用SentenceTransformer加载模型
        self.model = SentenceTransformer(
            model_name,
            device=device,
        )



    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Batch encode documents

        Args:
            texts:
                文档chunk列表

        Returns:
            embedding vectors
        """

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embeddings.tolist()


    def embed_query(self, query: str) -> List[float]:
        """
        Encode user query
        """

        embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        return embedding.tolist()
