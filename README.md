# AI Research Assistant

本地优先的科研 AI 助手：在通用能力层（LLM、Embedding、检索、Agent）之上，挂接领域知识库插件（如 NTC 论文 / 概念 / 公式）。硬件目标为 Mac + 约 24G 统一内存；先 CLI 跑通，再补 Web。

当前已落地：

- **LLM**：Ollama + 本地 `qwen3:14b`
- **Embedding**：`BAAI/bge-m3`（sentence-transformers）

检索、向量库、知识库插件与 Agent 仍在规划中，见 [docs/PLAN.md](docs/PLAN.md)。

## 当前结构

```text
src/ai_research_assistant/core/
├── llm/           # 大模型抽象与 Ollama 实现
└── embedding/     # 向量模型抽象与 BGE-M3 实现

tests/
├── test_llm.py
└── test_embedding.py
```

## 环境准备

需要：

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)
- 本地模型 `qwen3:14b`

安装 Python 依赖：

```bash
uv sync --dev
```

确认模型已安装：

```bash
ollama list
```

如果列表中没有 `qwen3:14b`，安装它：

```bash
ollama pull qwen3:14b
```

## 运行本地模型

测试前必须确保 Ollama 服务在运行。若 Ollama 桌面应用没有自动启动服务，请在一个独立终端执行：

```bash
ollama serve
```

默认服务地址是 `http://localhost:11434`。

## 测试

在另一个终端、项目根目录中运行。

LLM 冒烟测试（会打印模型回答）：

```bash
uv run python tests/test_llm.py
```

pytest 测试：

```bash
uv run pytest -q
```

仅运行 Embedding 集成测试：

```bash
uv run pytest -q tests/test_embedding.py
```

也可以运行会输出结果的 Embedding 冒烟脚本：

```bash
uv run python tests/test_embedding.py
```

预期结果：

```text
6 passed
```

## Qwen3 thinking 模式

默认关闭 thinking，以确保短回答能直接写入 `message.content`。如需开启，在创建配置时设置：

```python
from ai_research_assistant.core.llm.config import LLMConfig
from ai_research_assistant.core.llm.factory import create_llm

llm = create_llm(
    LLMConfig(
        model_name="qwen3:14b",
        think=True,
        max_tokens=4096,
    )
)
```

`max_tokens` 同时限制思考 token 和最终回答 token；开启 thinking 时不建议使用测试示例中的 `256`，建议至少设为 `2048`。

`generate()` / `chat()` 仍只返回最终回答；开启 `think=True` 后，最近一次思考文本会保存在 `llm.last_thinking`。运行 `tests/test_llm.py` 时会自动打印思考过程，关闭 thinking 则只打印最终回答。
