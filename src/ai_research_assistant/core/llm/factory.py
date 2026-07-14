"""
LLM factory.

Create LLM instances according to configuration.
"""

from .config import LLMConfig
from .base import BaseLLM
from .ollama_llm import OllamaLLM


# Registry of supported LLM providers.
# Mapping from provider name to the concrete implementation class.
_PROVIDERS: dict[str, type[BaseLLM]] = {
    "ollama": OllamaLLM,
}


def create_llm(
    config: LLMConfig
) -> BaseLLM:
    """
    Create an LLM instance.

    Args:
        config:
            LLM configuration.

    Returns:
        A concrete LLM implementation.

    Raises:
        ValueError:
            If the configured provider is not supported.
    """

    provider_cls = _PROVIDERS.get(config.provider)

    if provider_cls is None:
        raise ValueError(
            f"Unsupported LLM provider: {config.provider}"
        )

    return provider_cls(
        model_name=config.model_name,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        think=config.think,
    )
    
