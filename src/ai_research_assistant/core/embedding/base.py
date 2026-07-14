from abc import ABC, abstractmethod
from typing import List


class BaseEmbedding(ABC):
    """
    Embedding模型抽象基类

    所有embedding模型（BGE、Qwen embedding、OpenAI embedding等）
    都需要继承该类并实现对应方法。
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        将文本转换为向量

        Args:
            texts:
                输入文本列表，例如：
                [
                    "What is RAG?",
                    "RAG combines retrieval and generation."
                ]

        Returns:
            List[List[float]]:
                embedding向量列表，例如：
                [
                    [0.12, 0.34, ...],
                    [0.56, 0.78, ...]
                ]
        """
        pass


    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """
        将查询文本转换为向量

        用于检索阶段。

        Args:
            query:
                用户问题

        Returns:
            单个embedding向量
        """
        pass
