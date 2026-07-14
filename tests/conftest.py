import pytest

from ai_research_assistant.core.llm.config import LLMConfig
from ai_research_assistant.core.llm.factory import create_llm


@pytest.fixture
def llm():
    return create_llm(
        LLMConfig(
            provider="ollama",
            model_name="qwen3:14b",
            base_url="http://localhost:11434",
            temperature=0.7,
            max_tokens=2048,
            think=True,
        )
    )
