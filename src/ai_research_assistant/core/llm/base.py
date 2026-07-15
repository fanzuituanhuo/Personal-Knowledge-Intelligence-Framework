"""
Base interface for Large Language Models.

All LLM backends (Ollama, OpenAI, vLLM, etc.)
should inherit from BaseLLM.
"""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Abstract base class for language model providers.

    This class defines the unified interface that
    all LLM implementations should follow.
    """

    def __init__(
        self,
        model_name: str,
    ):
        """
        Initialize LLM.

        Args:
            model_name:
                Name of the language model.
                e.g. qwen3:8b, llama3.1, gpt-5
        """

        self.model_name = model_name


    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate text response from a prompt.

        Args:
            prompt:
                User input prompt.

            system_prompt:
                Optional system instruction.

            temperature:
                Sampling temperature.

            max_tokens:
                Maximum generated tokens.

        Returns:
            Generated text.
        """

        pass


    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """
        Chat-style generation.

        Args:
            messages:
                Conversation history.

                Example:
                [
                    {
                        "role": "system",
                        "content": "You are a research assistant."
                    },
                    {
                        "role": "user",
                        "content": "Explain VAE."
                    }
                ]

            temperature:
                Sampling temperature.

            max_tokens:
                Maximum output tokens.

        Returns:
            Assistant response.
        """

        pass
