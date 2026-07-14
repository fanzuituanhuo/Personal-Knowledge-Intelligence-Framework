"""
Embedding configuration module.

Centralize model and runtime settings.
"""

from dataclasses import dataclass

import torch
@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding models.
    """

    # 使用的后端
    provider: str = "bge"

    # 模型名称或本地路径
    model_name: str = "BAAI/bge-m3"

    # 推理设备：cpu / cuda / mps
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
