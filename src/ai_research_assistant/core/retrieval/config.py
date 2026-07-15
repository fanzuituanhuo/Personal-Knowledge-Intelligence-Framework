"""
Retriever configuration module.

Centralize retrieval backend settings.
"""

from dataclasses import dataclass


@dataclass
class RetrieverConfig:
    """
    Configuration for retrieval backends.
    """

    # 使用的后端
    provider: str = "dense"
