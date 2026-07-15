"""
Ollama LLM backend.

This module provides local LLM inference through Ollama,
such as Qwen3.5, Llama, DeepSeek, etc.
"""

import requests

from .base import BaseLLM
from .config import LLMConfig


class OllamaLLM(BaseLLM):
    """
    Local LLM implementation using Ollama.
    """

    def __init__(
        self,
        model_name: str = LLMConfig.model_name,
        base_url: str = LLMConfig.base_url,
        temperature: float = LLMConfig.temperature,
        max_tokens: int = LLMConfig.max_tokens,
        think: bool = LLMConfig.think,
    ):
        """
        Args:
            model_name:
                Ollama model name.

            base_url:
                Ollama server address.
        """

        super().__init__(model_name)

        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.think = think
        # 最近一次调用的思考文本；仅 think=True 时会填充。
        self.last_thinking: str | None = None


    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate response from a single prompt.
        """

        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )


    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Chat generation through Ollama API.
        """

        if temperature is None:
            temperature = self.temperature

        if max_tokens is None:
            max_tokens = self.max_tokens

        # ’这次要让模型怎么生成‘
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            # Qwen3 的思考 token 会计入 num_predict，短回答可能因此没有 content。
            "think": self.think,
            "options": {
                "temperature": temperature,
            },
        }


        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens


        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )


        response.raise_for_status()


        result = response.json()


        message = result.get("message", {})
        content = message.get("content", "")
        thinking = message.get("thinking") or ""

        self.last_thinking = thinking if self.think else None

        if not content.strip():
            raise RuntimeError(
                "Ollama 返回了空的 message.content；"
                "请检查模型是否耗尽 num_predict，或确认 Qwen3 thinking 已关闭。"
            )

        return content
