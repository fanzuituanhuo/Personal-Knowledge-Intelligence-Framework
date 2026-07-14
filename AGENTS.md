# AGENTS.md

给编码 Agent 的项目说明。人类入门见 [README.md](README.md)，完整规划见 [docs/PLAN.md](docs/PLAN.md)。

## 项目是什么

本地优先的科研 AI 助手：通用能力层（LLM / Embedding / Retrieval / Agent）+ 领域知识库插件（如 NTC）。硬件假设为 Mac + 约 24G 统一内存。

当前已落地：`core/llm`（Ollama + Qwen3）、`core/embedding`（BGE-M3）。后续按 `docs/PLAN.md` 阶段推进，不要跳阶段大改架构。

## 常用命令

```bash
uv sync --dev                          # 安装依赖（只用 uv，不要 pip install）
uv run pytest -q                       # 全量测试
uv run pytest -q tests/test_llm.py     # LLM 测试（需 ollama serve）
uv run pytest -q tests/test_embedding.py
uv run python tests/test_llm.py        # 带打印的冒烟脚本
uv run python tests/test_embedding.py
ollama serve                           # 本地 LLM 服务，默认 http://localhost:11434
```

加依赖用 `uv add <pkg>`，不要混用 pip。

## 架构约定

`core/` 能力模块统一三件套：

| 文件 | 职责 |
|------|------|
| `base.py` | `BaseXxx` 抽象接口 |
| `config.py` | `@dataclass` 配置（含 `provider`） |
| `factory.py` | `create_xxx(config)` + `_PROVIDERS` 注册表 |
| `<provider>_xxx.py` | 具体实现 |

参考实现：

- LLM：`src/ai_research_assistant/core/llm/`
- Embedding：`src/ai_research_assistant/core/embedding/`

新建能力时对齐上述模式，不要只写实现类而跳过 config/factory。

## 技术选型（默认）

| 组件 | 选择 |
|------|------|
| 包管理 | `uv` |
| Python | `>=3.12` |
| LLM | Ollama，`qwen3:14b` |
| Embedding | `BAAI/bge-m3` via `sentence-transformers` |
| 向量库 | ChromaDB |
| API | FastAPI（尚未实现） |

- 先本地模型，再考虑 API fallback。
- 先 CLI / 测试跑通，再补 Web。
- Embedding 设备默认可用 `mps`（Apple Silicon），否则 `cpu`。

## 命名

1. 抽象类：`BaseXxx`
2. 实现类：技术/后端命名，如 `OllamaLLM`、`BGEEmbedding`、`ChromaVectorStore`
3. Collection：`{domain}/{entity_type}`，如 `ntc/papers`
4. 配置字段：小写下划线；后续支持 `.env` / Pydantic Settings

## 编码原则

- 只改任务相关代码；不做顺手重构、不扩 scope。
- 新模块放 `src/ai_research_assistant/`，包名与目录一致。
- 接口返回类型清晰；LLM/Embedding 通过工厂创建，调用方依赖抽象基类。
- 每个核心类至少一个测试；冒烟脚本可 `python tests/test_*.py` 直接跑。
- 不要提交密钥、`.env`、大模型权重；模型用本机缓存或 Ollama，不进仓库。

## 验证「完成」

改动后至少：

1. 相关测试通过：`uv run pytest -q tests/test_<module>.py`
2. 若改了依赖：更新 `pyproject.toml`，并跑 `uv sync --dev`
3. 若改了对外用法：同步更新 `README.md` 中对应段落

## 不要做

- 不要用 `pip install` 往 uv 环境塞包
- 不要把 Ollama embedding 模型当成 `sentence-transformers` 的 `BAAI/bge-m3` 替代品混用
- 不要在未读 `docs/PLAN.md` 的情况下重写目录结构
- 不要默认开启 Qwen3 `think=True`（接口默认只要最终答案；开启时 `max_tokens` 需足够大）
