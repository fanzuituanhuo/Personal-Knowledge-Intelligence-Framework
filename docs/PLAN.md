# AI Research Assistant —— 构建计划

> 目标：从 0 搭建一个「通用科研能力框架 + 领域知识库插件」的本地 AI 助手。
> 硬件：Mac M5 Pro，24G 统一内存。
> 原则：先 CLI 跑通，再补 Web；模型调用以本地为主，同时支持外接云端 API（同一套 `BaseLLM` / 工厂接口切换）。

---

## 一、仓库顶层结构

```
Personal-Knowledge-Intelligence-Framework/   # GitHub 仓库根目录
│
├── README.md                   # 项目简介与快速开始
├── pyproject.toml              # 依赖、脚本、包配置
├── .env.example                # 环境变量模板
├── .gitignore
│
├── docs/                       # 项目文档
│   └── PLAN.md
│
├── scripts/                    # 一次性工具脚本
│
├── tests/                      # 测试
│   └── ...
│
└── src/                        # Python 源码
    └── ai_research_assistant/  # 主包
        │
        ├── core/               # 通用能力层
        │   ├── llm/
        │   ├── embedding/
        │   ├── retrieval/
        │   ├── memory/
        │   └── agent/
        │
        ├── knowledge/          # 领域知识库插件
        │   ├── ntc/
        │   │   ├── papers/
        │   │   ├── concepts/
        │   │   └── equations/
        │   ├── semantic_comm/
        │   └── topology_opt/
        │
        ├── database/           # 持久化层
        │   ├── vector_store/
        │   ├── relational/
        │   └── file_store/
        │
        ├── api/                # 后端服务接口
        │   ├── routers/
        │   └── models/
        │
        └── frontend/           # 前端界面
```

---

## 二、业务层次说明

### 2.1 `core/`：通用能力层

与具体研究领域无关，可被所有知识库复用。

| 模块 | 职责 |
|------|------|
| `llm/` | 大语言模型接口与实现：默认 Ollama 本地模型，也可外接 OpenAI 兼容等云端 API |
| `embedding/` | 文本向量模型接口与实现 |
| `retrieval/` | 语义检索、混合检索、重排序、上下文构造 |
| `memory/` | 对话历史、工作记忆、长期事实记忆 |
| `agent/` | Agent 编排、工具调用、任务规划 |

### 2.2 `knowledge/`：领域知识库插件

每个研究领域是一个插件包，内部按科研实体组织。

例如 `ntc/`：
- `papers/`：论文原文、摘要、方法、实验、结论
- `concepts/`：核心概念、定义、术语关系
- `equations/`：公式、推导、符号说明

每类实体可以独立配置：
- 解析方式
- 分块策略
- 元数据 schema
- 展示模板

### 2.3 `database/`：持久化层

统一封装所有数据存储：
- `vector_store/`：向量数据库（ChromaDB / Qdrant）
- `relational/`：关系型数据（SQLite，保存配置、元数据、对话历史）
- `file_store/`：原始文件存储（PDF / Markdown / 网页快照）

### 2.4 `api/` 与 `frontend/`

- `api/`：FastAPI 后端服务
- `frontend/`：Web 前端（先纯 HTML/JS，后续可升级框架）

---

## 三、分阶段计划

### 阶段 1：通用能力骨架

**目标**：搭出 `core/` + `database/` 的抽象与基础实现。

**要写的文件**：
- `pyproject.toml`
- `.env.example`
- `src/ai_research_assistant/__init__.py`
- `src/ai_research_assistant/core/llm/base.py`
- `src/ai_research_assistant/core/llm/ollama.py`
- `src/ai_research_assistant/core/embedding/base.py`
- `src/ai_research_assistant/core/embedding/sentence_transformers.py`
- `src/ai_research_assistant/core/retrieval/base.py`
- `src/ai_research_assistant/core/retrieval/simple_retriever.py`
- `src/ai_research_assistant/core/memory/base.py`
- `src/ai_research_assistant/core/memory/conversation_memory.py`
- `src/ai_research_assistant/core/agent/base.py`
- `src/ai_research_assistant/core/agent/react_agent.py`
- `src/ai_research_assistant/database/vector_store/base.py`
- `src/ai_research_assistant/database/vector_store/chroma_store.py`
- `src/ai_research_assistant/database/relational/base.py`
- `src/ai_research_assistant/database/config.py`

**架构能力点**：
1. 抽象类设计：为什么每个能力都要先定义 `BaseXxx`？
2. 依赖注入：配置如何贯穿所有模块？
3. 接口隔离：LLM、Embedding、Vector Store 各自独立

**验收标准**：
- 所有模块能正常 import
- 每个抽象类都有最小测试
- 能调用 Ollama 和本地 embedding 模型

---

### 阶段 2：文档解析与科研实体抽象

**目标**：把 PDF / 网页 / 笔记统一解析成科研实体。

**要写的文件**：
- `src/ai_research_assistant/core/document/base.py`：`Document`, `Chunk`
- `src/ai_research_assistant/core/document/loaders/pdf.py`
- `src/ai_research_assistant/core/document/loaders/web.py`
- `src/ai_research_assistant/core/document/loaders/markdown.py`
- `src/ai_research_assistant/core/document/chunker.py`
- `src/ai_research_assistant/knowledge/base.py`：`KnowledgeBase`, `EntityType`
- `src/ai_research_assistant/knowledge/ntc/schema.py`

**架构能力点**：
1. 文档模型：如何保留来源、页码、章节、公式编号等元数据？
2. 领域 Schema：不同研究方向的数据结构如何定义？
3. 分块策略：论文、概念、公式分别怎么切分？

**验收标准**：
- 能解析 PDF / Markdown / 网页
- 能按领域 schema 存入向量库
- 元数据完整保留

---

### 阶段 3：多知识库管理

**目标**：实现 `knowledge/<domain>/` 的创建、导入、检索。

**要写的文件**：
- `src/ai_research_assistant/knowledge/manager.py`
- `src/ai_research_assistant/knowledge/ntc/papers/`
- `src/ai_research_assistant/knowledge/ntc/concepts/`
- `src/ai_research_assistant/knowledge/ntc/equations/`
- `src/ai_research_assistant/knowledge/semantic_comm/`
- `src/ai_research_assistant/knowledge/topology_opt/`

**架构能力点**：
1. 插件机制：如何新增一个研究领域？
2. collection 命名：`ntc/papers`、`ntc/concepts` 如何映射到向量库？
3. 跨库检索：能否同时搜多个领域？

**验收标准**：
- 能创建多个领域知识库
- 每个领域下能创建 papers / concepts / equations
- 支持跨库搜索

---

### 阶段 4：RAG 与对话

**目标**：把检索和 LLM 串起来，实现可溯源问答。

**要写的文件**：
- `src/ai_research_assistant/core/retrieval/rag_engine.py`
- `src/ai_research_assistant/core/agent/research_agent.py`
- `src/ai_research_assistant/core/prompts/rag.py`
- `src/ai_research_assistant/api/routers/chat.py`

**架构能力点**：
1. RAG 流程：query → 意图识别 → 检索 → 重排序 → 构造上下文 → LLM
2. 引用溯源：答案如何标注来源文献？
3. 流式输出：如何减少等待感？

**验收标准**：
- 导入论文后能问答
- 回答带引用
- 支持流式输出

---

### 阶段 5：科研 Agent

**目标**：不只是问答，而是能做科研任务。

**功能**：
- 文献综述生成
- 论文精读（摘要 / 方法 / 实验 / 局限）
- 概念解释与公式推导
- arXiv / Semantic Scholar 搜索与导入

**要写的文件**：
- `src/ai_research_assistant/core/agent/tools/search_arxiv.py`
- `src/ai_research_assistant/core/agent/tools/read_paper.py`
- `src/ai_research_assistant/core/agent/tools/summarize.py`
- `src/ai_research_assistant/core/agent/research_assistant.py`

**验收标准**：
- Agent 能调用多个工具完成复杂任务
- 能生成带引用的综述

---

### 阶段 6：API 与前端

**目标**：把系统暴露为 Web 服务。

**要写的文件**：
- `src/ai_research_assistant/api/main.py`
- `src/ai_research_assistant/api/routers/knowledge.py`
- `src/ai_research_assistant/api/routers/chat.py`
- `src/ai_research_assistant/api/routers/documents.py`
- `src/ai_research_assistant/frontend/`：简单前端

**验收标准**：
- 能网页上传文档、管理知识库
- 能网页聊天，看到流式回答

---

## 四、技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| 语言 | Python 3.11+ | 生态最成熟 |
| 包管理 | `uv` | 已安装，速度快 |
| 配置 | Pydantic Settings | 类型安全，环境变量友好 |
| PDF 解析 | PyMuPDF / marker | 准确、可提取页码 |
| 网页解析 | crawl4ai / jina reader | 获取正文干净 |
| Embedding | BAAI/bge-m3 | 多语言，24G 内存可跑 |
| 向量库 | ChromaDB | 轻量，按 collection 隔离 |
| LLM（本地） | Ollama | Mac 最方便，默认路径 |
| LLM（外接） | OpenAI 兼容 API | 需要更强模型或本机资源不够时切换；密钥走 `.env` |
| 推荐本地模型 | qwen3:14b 等 | 中英文平衡，适配约 24G 统一内存 |
| API | FastAPI | 异步、文档自动生成 |
| 前端 | 先用纯 HTML + JS | 快速验证 |
| 测试 | pytest | 标准选择 |

---

## 五、命名约定

1. 抽象类：`BaseXxx`
2. 实现类：用技术命名，如 `OllamaLLM`、`OpenAILLM`、`ChromaVectorStore`
3. 知识库 collection 命名：`{domain}/{entity_type}`，例如 `ntc/papers`
4. 配置项：小写下划线，支持 `.env`
5. 每个核心类至少一个测试

---

## 六、下一步

从 **阶段 1：通用能力骨架** 开始，先写出 `core/` 和 `database/` 的抽象与最小实现。
