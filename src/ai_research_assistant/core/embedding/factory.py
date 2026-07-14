"""
Embedding factory.

Create embedding instances according to configuration.
"""

from .config import EmbeddingConfig
from .base import BaseEmbedding
from .bge_embedding import BGEEmbedding


# Registry of supported embedding providers.
# Mapping from provider name to the concrete implementation class.
_PROVIDERS: dict[str, type[BaseEmbedding]] = {
    "bge": BGEEmbedding,
}


def create_embedding(
    config: EmbeddingConfig,
) -> BaseEmbedding:
    """
    Create an embedding instance.

    Args:
        config:
            Embedding configuration.

    Returns:
        A concrete embedding implementation.

    Raises:
        ValueError:
            If the configured provider is not supported.
    """

    provider_cls = _PROVIDERS.get(config.provider)

    if provider_cls is None:
        raise ValueError(
            f"Unsupported embedding provider: {config.provider}"
        )

    return provider_cls(
        model_name=config.model_name,
        device=config.device,
    )
