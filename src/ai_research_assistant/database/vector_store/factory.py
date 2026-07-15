from ai_research_assistant.core.embedding.base import BaseEmbedding
from ai_research_assistant.database.vector_store.base import BaseVectorStore
from ai_research_assistant.database.vector_store.faiss_store import FAISSVectorStore

def create_vector_store(embedding: BaseEmbedding) -> BaseVectorStore:
    """
    Create a vector store instance.

    Args:
        embedding:
            The embedding model to use for vector store.

    Returns:
        A vector store instance using the embedding model.
    """
    return FAISSVectorStore(embedding.dimension)

