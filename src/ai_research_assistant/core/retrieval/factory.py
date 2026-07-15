"""
Retriever factory.

Create retriever instances according to configuration.
"""

from ai_research_assistant.database.vector_store.base import BaseVectorStore

from .base import BaseRetriever
from .config import RetrieverConfig
from .dense_retriever import DenseRetriever


# Registry of supported retriever providers.
# Mapping from provider name to the concrete implementation class.
_PROVIDERS: dict[str, type[BaseRetriever]] = {
    "dense": DenseRetriever,
}


def create_retriever(
    config: RetrieverConfig,
    vector_store: BaseVectorStore,
) -> BaseRetriever:
    """
    Create a retriever instance.

    Args:
        config:
            Retriever configuration.

        vector_store:
            Vector store used by dense retrieval backends.

    Returns:
        A concrete retriever implementation.

    Raises:
        ValueError:
            If the configured provider is not supported.
    """
    provider_cls = _PROVIDERS.get(config.provider)

    if provider_cls is None:
        raise ValueError(
            f"Unsupported retriever provider: {config.provider}"
        )

    return provider_cls(vector_store=vector_store)
