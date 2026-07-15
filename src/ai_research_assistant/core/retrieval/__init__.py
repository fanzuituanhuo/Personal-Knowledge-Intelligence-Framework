"""Retrieval abstractions and implementations."""

from .base import BaseRetriever
from .config import RetrieverConfig
from .dense_retriever import DenseRetriever
from .factory import create_retriever
from .semantic_retriever import SemanticRetriever

__all__ = [
    "BaseRetriever",
    "DenseRetriever",
    "RetrieverConfig",
    "SemanticRetriever",
    "create_retriever",
]
