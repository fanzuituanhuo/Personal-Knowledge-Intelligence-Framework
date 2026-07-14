"""
LLM configuration module.

Centralize model and inference settings.
"""

from dataclasses import dataclass


@dataclass
class LLMConfig:
    """
    Configuration for LLM inference.
    """

    # 使用的后端
    provider: str = "ollama"

    # 模型名称
    model_name: str = "qwen3:14b"

    # Ollama地址
    base_url: str = "http://localhost:11434"

    # 生成参数
    temperature: float = 0.7

    # 最大生成token数
    max_tokens: int = 2048

    # Qwen3 默认会输出 thinking；科研助手接口默认只返回最终答案。
    think: bool = False
