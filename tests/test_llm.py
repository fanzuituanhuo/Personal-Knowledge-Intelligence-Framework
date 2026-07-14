"""
Smoke test for local Ollama LLM (qwen3:14b).

Usage:
    uv run python tests/test_llm.py
"""

from ai_research_assistant.core.llm.config import LLMConfig
from ai_research_assistant.core.llm.factory import create_llm


def _print_reply(llm, reply: str) -> None:
    """think 模式打印思考过程，否则只打印最终回答。"""
    if getattr(llm, "think", False):
        thinking = getattr(llm, "last_thinking", None)
        print("--- thinking ---")
        print(thinking.strip() if thinking and thinking.strip() else "(无思考内容)")
        print("--- content ---")
    print(reply)
    print()


def test_generate(llm) -> None:
    print("=" * 50)
    print("[generate]")
    print("=" * 50)

    # thinking 与回答共用 max_tokens；开启时需要更大额度。
    max_tokens = 2048 if getattr(llm, "think", False) else 256

    reply = llm.generate(
        prompt="用一句话解释什么是变分自编码器（VAE）。",
        system_prompt="你是一个简洁的科研助手。",
        temperature=0.3,
        max_tokens=max_tokens,
    )

    _print_reply(llm, reply)
    assert isinstance(reply, str) and reply.strip(), "generate() 返回为空"


def test_chat(llm) -> None:
    print("=" * 50)
    print("[chat]")
    print("=" * 50)

    max_tokens = 2048 if getattr(llm, "think", False) else 256

    reply = llm.chat(
        messages=[
            {
                "role": "system",
                "content": "你是一个简洁的科研助手。",
            },
            {
                "role": "user",
                "content": "VAE 里的 KL 散度起什么作用？一句话回答。",
            },
        ],
        temperature=0.3,
        max_tokens=max_tokens,
    )

    _print_reply(llm, reply)
    assert isinstance(reply, str) and reply.strip(), "chat() 返回为空"


def main() -> None:
    config = LLMConfig(
        provider="ollama",
        model_name="qwen3:14b",
        base_url="http://localhost:11434",
        temperature=0.7,
        max_tokens=2048,
        think=True,
    )

    print(f"provider : {config.provider}")
    print(f"model    : {config.model_name}")
    print(f"base_url : {config.base_url}")
    print(f"think    : {config.think}")
    print()

    llm = create_llm(config)

    test_generate(llm)
    test_chat(llm)

    print("全部通过。")


if __name__ == "__main__":
    main()
