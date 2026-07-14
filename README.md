# Atlas

**Personal Knowledge Intelligence Framework**

本地优先的个人知识智能框架：在通用能力层之上挂接领域知识库，再长出具体助手——可以是科研助手，也可以是一门课的学习助手。

> 这是**应用侧的能力底座**（Framework），不是算力/模型服务意义上的 AI Infrastructure。  
> 不依赖 LangChain、LlamaIndex 等编排库；`core/` 自研薄抽象，直接对接后端。

## 它在做什么

Atlas 把「可复用的智能能力」和「可替换的知识域」拆开：

| 层 | 职责 | 例子 |
|----|------|------|
| **能力层 `core/`** | LLM、Embedding、检索、记忆、Agent | 换模型、换检索策略，场景代码尽量不动 |
| **知识层 `knowledge/`** | 按领域插件组织实体与 schema | 科研：论文 / 概念 / 公式；课程：讲义 / 概念 / 习题 |
| **场景** | 挂在框架上的具体用法 | 读论文问答、跟一门课复习对话 |

模型调用**默认本地**（Ollama 等），也支持**外接云端 API**（OpenAI 兼容等），经同一套 `BaseLLM` + factory 切换。

## 设计取舍

- **自研抽象，不用 LangChain / LlamaIndex**：接口形状是 `BaseXxx` + `config` + `factory`，依赖少、行为可控，便于按阶段扩展。
- **知识可插拔**：新领域 = 新插件，而不是另起一个 RAG demo。
- **先 CLI / 测试，再 API / Web**：先把本地链路跑通，再补服务与界面。
- **硬件假设**：Mac + 约 24G 统一内存。

完整分阶段计划见 [docs/PLAN.md](docs/PLAN.md)。

## 当前进度

已落地：

- **LLM**：Ollama + 本地 `qwen3:14b`（外接 API 实现待补）
- **Embedding**：`BAAI/bge-m3`（sentence-transformers）

规划中：检索、向量库、知识库插件、记忆、Agent、API / 前端。

> GitHub 仓库：`Personal-Knowledge-Intelligence-Framework`。Python 包名暂为 `ai-research-assistant` / `ai_research_assistant`，后续会与 Atlas 品牌对齐。

## 当前结构

```text
src/ai_research_assistant/core/
├── llm/           # 大模型抽象与 Ollama 实现
└── embedding/     # 向量模型抽象与 BGE-M3 实现

tests/
├── test_llm.py
└── test_embedding.py
```

## 快速开始

需要：

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/)
- 本地模型 `qwen3:14b`

```bash
uv sync --dev
ollama pull qwen3:14b
ollama serve   # 若桌面端未自动起服务；默认 http://localhost:11434
```

### 测试

```bash
# LLM 冒烟（打印模型回答）
uv run python tests/test_llm.py

# Embedding 冒烟
uv run python tests/test_embedding.py

# 全量 pytest
uv run pytest -q
```

预期：`6 passed`。

### 最小调用示例

```python
from ai_research_assistant.core.llm.config import LLMConfig
from ai_research_assistant.core.llm.factory import create_llm

llm = create_llm(LLMConfig(model_name="qwen3:14b"))
print(llm.generate("用一句话解释什么是知识增强。"))
```

## Qwen3 thinking 模式

默认关闭 thinking，短回答会直接写入 `message.content`。开启时：

```python
llm = create_llm(
    LLMConfig(
        model_name="qwen3:14b",
        think=True,
        max_tokens=4096,  # 需同时覆盖思考与最终回答；勿用过小值如 256
    )
)
```

`generate()` / `chat()` 仍只返回最终回答；思考文本在 `llm.last_thinking`。`tests/test_llm.py` 在开启 thinking 时会打印思考过程。

## 路线图（摘要）

1. 通用能力骨架（LLM / Embedding / 检索 / 记忆 / Agent + 存储）
2. 文档解析与领域实体 schema
3. 多知识库插件（科研、课程等）
4. Agent 编排与工具调用
5. API 与简易前端

细节与验收标准见 [docs/PLAN.md](docs/PLAN.md)。
