"""Integration tests for the local BGE-M3 embedding model.

Usage:
    uv run python tests/test_embedding.py
    uv run pytest -q tests/test_embedding.py
"""

import pytest

from ai_research_assistant.core.embedding.config import EmbeddingConfig
from ai_research_assistant.core.embedding.factory import create_embedding


MODEL_NAME = "BAAI/bge-m3"
EXPECTED_DIMENSION = 1024


@pytest.fixture(scope="module")
def embedding():
    """Load the model once for the whole embedding test module."""
    return create_embedding(
        EmbeddingConfig(
            provider="bge",
            model_name=MODEL_NAME,
        )
    )


def test_model_loads(embedding) -> None:
    """The configured SentenceTransformer model can be initialized."""
    assert embedding.model is not None
    assert embedding.model_name == MODEL_NAME


def test_texts_convert_to_vectors(embedding) -> None:
    """A batch of documents produces one non-empty vector per document."""
    texts = [
        "变分自编码器是一种生成模型。",
        "检索增强生成结合了信息检索与语言生成。",
    ]

    vectors = embedding.embed(texts)

    assert isinstance(vectors, list)
    assert len(vectors) == len(texts)
    assert all(isinstance(vector, list) and vector for vector in vectors)
    assert all(isinstance(value, float) for vector in vectors for value in vector)


def test_embedding_dimension_is_correct(embedding) -> None:
    """BGE-M3 document embeddings have the documented 1024 dimensions."""
    vector = embedding.embed(["用于检查向量维度的文本。"])[0]

    assert len(vector) == EXPECTED_DIMENSION


def test_query_embedding(embedding) -> None:
    """A retrieval query can be embedded into a non-empty BGE-M3 vector."""
    vector = embedding.embed_query("什么是 VAE 中的 KL 散度？")

    assert isinstance(vector, list)
    assert vector
    assert len(vector) == EXPECTED_DIMENSION
    assert all(isinstance(value, float) for value in vector)


def _print_device(embedding) -> None:
    """Print the configured and actual runtime device."""
    configured = embedding.device
    actual = getattr(embedding.model, "device", configured)
    print(f"Embedding device: configured={configured}, actual={actual}")


def test_device_is_reported(embedding) -> None:
    """The embedding instance exposes the configured device."""
    assert embedding.device in {"cpu", "cuda", "mps"}


def main() -> None:
    """Run the same checks as a standalone smoke test."""
    instance = create_embedding(
        EmbeddingConfig(
            provider="bge",
            model_name=MODEL_NAME,
            device=EmbeddingConfig.device, #按照config配置的device来加载模型
        )
    )

    _print_device(instance)

    test_model_loads(instance)
    test_texts_convert_to_vectors(instance)
    test_embedding_dimension_is_correct(instance)
    test_query_embedding(instance)
    test_device_is_reported(instance)

    print("Embedding 测试全部通过。")


if __name__ == "__main__":
    main()
