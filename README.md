# 🎭 RAG知识库智能平台（戏曲科普）

基于 **LangChain + LangGraph + Chroma + BM25 + FastAPI + Streamlit + Ollama** 的一体化 RAG 智能平台，融合 **知识问答、戏曲文献生成、检索指标评测** 与 **戏曲科普互动** 功能。

## ✨ 核心功能概览

| 模块 | 功能说明 |
|------|----------|
| 💬 **智能对话** | 普通单 Agent 问答 / 多智能体协作，支持反思重试机制 |
| 📚 **知识库管理** | 文档上传入库（txt/md/pdf）/ 混合检索测试 / 统计查看 |
| 🎭 **戏曲文献生成** | 一键生成戏曲学术文献，支持 **txt / pdf / md** 三种格式 |
| 📊 **指标评测** | 召回率 Recall@K / 命中率 HitRate@K / MRR@K / NDCG@K 量化测试 |
| 📜 **戏词解剖室** | 经典戏词逐句白话翻译 / 典故考据 / 人物心境 / 唱腔段式标注 |
| 🎭 **戏中人对谈** | 与穆桂英、白素贞、曹操等戏曲人物人格化对话 |
| 🎮 **知识闯关** | 游戏化出题 / 判题讲解，支持多主题 × 多档难度 |
| 🧭 **个性化学戏路线** | 按剧种/天数生成阶梯课程，支持进度追踪 |
| 🎨 **脸谱画像** | 生成专属戏曲人格脸谱解读，支持即梦AI文生图（可选） |
| ⚙️ **系统管理** | 健康检查 / 缓存管理 / 日志级别动态调整 |

## 📁 项目目录结构

```
agent-study-rag/
├── api/                          # FastAPI 后端
│   ├── main.py                   # 应用入口（生命周期/异常处理/限流/路由注册）
│   ├── schema.py                 # Pydantic 请求/响应模型
│   ├── limiter.py                # 全局限速器
│   ├── stream_response.py        # SSE 流式响应封装
│   └── routes/                   # 业务路由模块
│       ├── chat_routes.py        # 对话接口（普通/多Agent/流式）
│       ├── kb_routes.py          # 知识库管理接口（入库/检索/统计/清空）
│       ├── literature_routes.py  # 戏曲文献生成接口
│       ├── evaluation_routes.py  # 检索指标评测接口
│       ├── opera_routes.py       # 戏曲科普五大功能接口
│       └── system_routes.py      # 系统管理接口（健康/缓存/日志）
│
├── agent/                        # Agent 智能体
│   ├── graph_base.py             # 单 Agent（工具规划-执行-反思循环）
│   ├── multi_agent.py            # 多 Agent（主管-角色化 Worker 协作模式）
│   ├── intent.py                 # 意图识别
│   ├── intent_guard.py           # 意图守卫（安全校验）
│   ├── llm_utils.py              # LLM 调用工具（重试/超时）
│   ├── output_validator.py       # 输出校验器
│   ├── task_context.py           # 任务上下文管理
│   ├── trace.py                  # Agent 调用可观测性
│   ├── session_memory.py         # Redis 会话记忆
│   └── memory/                   # 分层隔离记忆模块
│       ├── permanent_memory.py   # 永久静态记忆
│       ├── task_memory.py        # 任务级记忆
│       ├── session_memory.py     # 会话时序记忆
│       ├── unified_memory.py     # 统一认知记忆抽象
│       └── memory_manager.py     # 统一记忆管理器
│
├── opera/                        # 戏曲科普业务模块
│   ├── lyrics.py                 # 戏词解剖室（逐句译注/典故/唱腔/心境）
│   ├── character.py              # 戏中人对谈（角色人格化对话）
│   ├── quiz.py                   # 知识闯关（游戏化出题/判题/讲解）
│   ├── guide.py                  # 个性化学戏路线（阶梯课程/进度追踪）
│   └── face.py                   # 脸谱画像（LLM 设计 + 即梦AI文生图）
│
├── rag/                          # RAG 检索增强生成核心
│   ├── loader/                   # 文档加载器（txt/md/pdf）
│   ├── splitter/                 # 文本分块器
│   ├── chain/                    # RAG 链
│   ├── reranker/                 # LLM Reranker 精排
│   ├── agentic/                  # Agentic RAG（查询改写/文档评分/答案评分/自动检索）
│   └── vectorstore/              # Chroma 向量库 + BM25 关键词索引 + 混合检索
│
├── literature/                   # 戏曲文献生成模块
│   └── generator.py              # 文献生成器（txt/pdf/md 输出）
│
├── evaluation/                   # 检索指标评测模块
│   ├── metrics_evaluator.py      # 基础指标（Recall/HitRate/MRR/NDCG）
│   ├── advanced_evaluator.py     # 高级指标（Faithfulness/AnswerRelevance 等）
│   ├── dynamic_questions.py      # 动态问题生成器
│   ├── eval_agent.py             # 评测 Agent
│   ├── eval_prompts.py           # 评测提示词模板
│   └── pre_filter.py             # 评测前置过滤器
│
├── mcp_server/                   # MCP Server
│   └── rag_tools_mcp.py          # JSON-RPC 工具服务（kb_search/doc_get/lit_generate/eval_rag）
│
├── tools/                        # Agent 工具注册表
│   └── custom_tools.py           # 知识库混合检索工具
│
├── frontend/                     # Streamlit 前端
│   ├── app.py                    # 首页（核心功能卡片 + 知识库概览）
│   ├── api_client.py             # 后端 API 客户端
│   └── pages/                    # 多页面
│       ├── 1_💬_智能对话.py
│       ├── 2_📚_知识库管理.py
│       ├── 3_🎭_戏曲文献生成.py
│       ├── 4_📊_指标评测.py
│       ├── 5_📜_戏词解剖室.py
│       ├── 6_🎭_戏中人对谈.py
│       ├── 7_🎮_知识闯关.py
│       ├── 8_🧭_个性化学戏路线.py
│       └── 9_🎨_脸谱画像.py
│
├── knowledge_base/opera/         # 戏曲领域知识库文档（多剧种/流派）
├── utils/                        # 工具库
│   ├── logger.py                 # 日志系统（控制台 + 文件滚动）
│   ├── rag_exceptions.py         # 统一异常体系
│   ├── exception_handler.py      # 全局异常装饰器
│   ├── cache_utils.py            # 双层 Redis 缓存
│   ├── redis_client.py           # Redis 客户端
│   └── json_repair.py            # JSON 修复工具
│
├── tests/                        # 测试套件
│   ├── test_reranker.py
│   ├── test_agentic.py
│   ├── test_trace.py
│   ├── test_mcp.py
│   ├── test_compound_intent.py
│   ├── test_image_scheduling.py
│   ├── test_new_bugs.py
│   └── test_three_fixes.py
│
├── memory_store/                 # 分层记忆存储目录
│   ├── permanent_memory.json     # 永久静态记忆（全局共享）
│   ├── tasks/                    # 任务级记忆（每个任务独立文件）
│   ├── sessions/                 # 会话时序记忆
│   └── archives/                 # 会话归档（自动清理）
│
├── config.py                     # 全局配置文件
├── pyproject.toml                # 项目元数据与依赖声明
├── requirements.txt              # pip 依赖列表
├── start_backend.py              # 后端启动脚本
├── start_frontend.py             # 前端启动脚本
├── generate_opera_kb.py          # 工具脚本：批量生成戏曲知识库文档并入库
├── ARCHITECTURE.md               # 系统架构说明文档
└── README.md                     # 本文件
```

## 🌍 环境依赖

| 依赖 | 说明 | 安装方式 |
|------|------|----------|
| **Python** | ≥ 3.11 | [python.org](https://python.org) |
| **Ollama** | 本地 LLM 运行环境 | [ollama.ai](https://ollama.ai) |
| **Redis** | 缓存与会话存储 | 系统包管理器或 [redis.io](https://redis.io) |
| **uv**（推荐） | Python 包管理器 | `pip install uv` |

## 🚀 部署启动步骤

### 1. 准备 Ollama 模型

```bash
# 启动 Ollama 服务后，拉取所需模型
ollama pull qwen2.5:3b    # LLM 对话模型（支持长上下文）
ollama pull bge-m3         # 嵌入模型（中文检索优化）
```

### 2. 启动 Redis

```bash
# Linux / macOS
redis-server

# Windows
redis-server.exe
```

### 3. 安装项目依赖

**推荐使用 uv（更快、更可靠）：**

```bash
# 在项目根目录下执行
uv sync
```

**或使用 pip：**

```bash
pip install -r requirements.txt
```

> `volcengine-python-sdk` 为可选依赖，用于脸谱画像的即梦 AI 图像生成。未安装或未配置 AK/SK 时，自动降级为纯文本版脸谱画像，不影响其他功能。

### 4. 启动后端服务

```bash
python start_backend.py
```

启动后自动完成：
- Redis 连接初始化
- BM25 关键词索引重建
- 单 Agent / 多 Agent 实例构建
- 文献输出目录创建
- 分层记忆模块初始化（自动清理过期会话）

API 文档地址：http://127.0.0.1:8000/docs

### 5. 启动前端

```bash
python start_frontend.py
```

访问地址：http://localhost:8501

## ⚙️ 环境变量配置清单

所有配置集中在 `config.py`，无需额外 `.env` 文件。主要配置项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LLM_MODEL` | `qwen2.5:3b` | Ollama 对话模型名称 |
| `EMBED_MODEL` | `bge-m3` | Ollama 嵌入模型名称 |
| `LLM_TEMP` | `0.1` | LLM 生成温度 |
| `LLM_TIMEOUT` | `120` | LLM 调用超时（秒） |
| `CHUNK_SIZE` | `120` | 文档分块大小 |
| `CHUNK_OVERLAP` | `15` | 分块重叠量 |
| `CHROMA_PERSIST_PATH` | `./chroma_db` | 向量库持久化路径 |
| `RETRIEVE_TOP_K` | `3` | 默认检索返回条数 |
| `ENABLE_HYBRID_SEARCH` | `True` | 是否启用混合检索（BM25+向量） |
| `ENABLE_RERANKER` | `True` | 是否启用 LLM 精排 |
| `ENABLE_RAG_CACHE` | `True` | 是否启用检索缓存 |
| `MAX_REFLECT_TIMES` | `2` | Agent 最大反思重试次数 |
| `JIMENG_IMAGE_ENABLED` | `False` | 是否启用即梦 AI 图像生成 |
| `JIMENG_ACCESS_KEY_ID` | `""` | 火山引擎 AccessKey ID |
| `JIMENG_SECRET_ACCESS_KEY` | `""` | 火山引擎 Secret Access Key |
| `LOG_LEVEL` | `INFO` | 日志级别（DEBUG/INFO/WARNING/ERROR） |
| `LOG_TO_FILE` | `True` | 是否将日志写入文件 |
| `LOG_SAVE_PATH` | `./logs` | 日志文件存放目录 |
| `LITERATURE_ROOT_DIR` | `./literature_output` | 文献输出根目录 |
| `EVAL_DYNAMIC_QUESTION_COUNT` | `6` | 动态评测生成问题数量 |
| `EVAL_REPORT_PATH` | `./evaluation_report/report.md` | 评测报告保存路径 |

## 🔌 API 接口总览

### 对话接口 `/api/chat/*`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat/normal` | 普通 Agent 对话 |
| POST | `/api/chat/multi_agent` | 多智能体协作对话 |
| POST | `/api/chat/stream` | 流式对话（SSE） |

### 知识库接口 `/api/kb/*`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/kb/ingest` | 文档按路径入库 |
| POST | `/api/kb/upload` | 文件上传入库 |
| POST | `/api/kb/retrieve` | 混合检索测试 |
| GET  | `/api/kb/stats` | 知识库统计 |
| POST | `/api/kb/clear` | 清空知识库 |

### 戏曲文献接口 `/api/literature/*`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/literature/generate` | 生成戏曲文献（txt/pdf/md） |
| GET  | `/api/literature/list` | 文献文件列表 |
| GET  | `/api/literature/read` | 读取文献内容 |
| GET  | `/api/literature/download` | 下载文献文件 |

### 评测接口 `/api/evaluation/*`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/evaluation/run` | 执行检索指标评测 |
| GET  | `/api/evaluation/report` | 读取评测报告 |

### 戏曲科普接口 `/api/opera/*`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/opera/lyrics/annotate` | 戏词解剖（逐句翻译/典故/心境/唱腔） |
| GET  | `/api/opera/character/list` | 戏曲人物列表 |
| POST | `/api/opera/character/chat` | 与戏曲人物对谈 |
| POST | `/api/opera/quiz/generate` | 知识闯关出题 |
| POST | `/api/opera/quiz/topics` | 可用出题主题 |
| POST | `/api/opera/quiz/check` | 判题并返回讲解 |
| POST | `/api/opera/course/generate` | 生成学戏课程路线 |
| POST | `/api/opera/course/progress` | 查询学戏进度 |
| POST | `/api/opera/face/generate` | 生成专属脸谱画像 |

### 系统接口 `/api/*`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/api/health` | 健康检查 |
| POST | `/api/cache/clear` | 清空缓存 |
| GET  | `/api/cache/status` | 缓存状态 |
| POST | `/api/log/level` | 日志级别动态调整 |

## 🧩 功能模块说明

### 智能对话

- **单 Agent**：基于 LangGraph 的 ReAct 模式，自动规划工具调用（知识库检索），支持反思重试。
- **多 Agent**：主管-角色化 Worker 协作模式，主管识别意图后分派给 6 个角色化 Worker 执行。

### 知识库管理

- 支持 txt / md / pdf 文档上传，自动分块、向量化入库。
- 检索采用 **BM25 关键词 + 向量语义** 混合检索，分数融合后可选 LLM Reranker 精排。
- 提供统计接口查看向量库和 BM25 索引状态。

### 戏曲文献生成

- 按流派（如京剧）和主题一键生成戏曲学术文献。
- 支持 **txt**（纯文本）、**pdf**（带排版）、**md**（Markdown）三种输出格式。
- 文献统一存放至 `literature_output/` 目录，按格式分子文件夹。

### 检索指标评测

- 指标：Recall@K / HitRate@K / MRR@K / NDCG@K。
- 支持动态问题生成：根据传入主题 + 知识库内容自动生成评测问题集。
- 多次运行取均值，生成 Markdown 报告。

### 戏词解剖室

- 输入经典戏词，AI 输出结构化解读：逐句白话翻译、典故考据、人物心境、唱腔段式标注。
- 基于知识库混合检索增强考据准确性。

### 戏中人对谈

- 内置穆桂英、白素贞、曹操等多名戏曲角色，每个角色有独立性格、口吻、背景。
- 以角色口吻进行人格化对话，自动检索知识库增强回答质量。

### 知识闯关

- 多主题 × 多难度档位，游戏化出题、判题、知识点讲解。
- 支持进度激励。

### 个性化学戏路线

- 按剧种 + 天数生成阶梯式学戏课程，每天包含标题、学习内容、互动问题。
- 课程进度自动记录，支持进度查询。

### 脸谱画像

- LLM 根据用户偏好生成专属脸谱设计（名号/主色/图案/气质匹配/人格解读）。
- 可选：配置火山引擎即梦 AI 后自动生成脸谱图像（`JIMENG_IMAGE_ENABLED=True`）。
- 未配置时自动降级为纯文本版，不影响功能。

## 🧠 分层记忆系统

Agent 长时间记忆采用**分层隔离存储**设计：

| 层级 | 存储位置 | 说明 |
|------|----------|------|
| 永久静态记忆 | `memory_store/permanent_memory.json` | 系统规则、领域规范、用户偏好，全局共享 |
| 任务级记忆 | `memory_store/tasks/{task_id}.json` | 每个任务独立空间，支持父子任务嵌套 |
| 会话时序记忆 | `memory_store/sessions/{session_id}.json` | 对话轨迹、工具调用记录，支持自动归档与过期清理 |

## 🎭 多智能体协作

`agent/multi_agent.py` 采用 **主管-工人** 模式：

| Worker | 职责 | 关联功能 |
|--------|------|----------|
| `search_worker` | 知识库混合检索 | 戏曲知识问答 |
| `lyrics_worker` | 戏词翻译/典故考据 | 戏词解剖室 |
| `character_worker` | 角色人格化对话 | 戏中人对谈 |
| `quiz_worker` | 主题出题 | 知识闯关 |
| `guide_worker` | 阶梯学戏课程 | 个性化学戏路线 |
| `face_worker` | 脸谱画像生成 | 脸谱画像 |

## 🛠️ 工程化特性

- **统一异常体系**：所有业务异常继承 `BaseRAGException`，含错误码/错误信息/原始异常。
- **全局异常捕获**：FastAPI 全局异常处理器 + 装饰器双重保障。
- **结构化日志**：控制台 + 文件滚动输出，日志级别可动态调整。
- **双层缓存**：检索缓存 + 问答缓存，Redis 实现，可开关控制。
- **限流保护**：所有接口 IP 限流。
- **模块化路由**：按业务域拆分路由，代码结构清晰。
- **安全校验**：文献文件访问路径白名单校验，防目录穿越。
- **MCP Server**：JSON-RPC over stdio 方式暴露核心工具。
- **Agent 可观测性**：自动记录调用步骤、耗时、工具调用、错误，支持统计汇总。

## 🔧 工具脚本

### 生成戏曲知识库

```bash
# 批量生成多剧种戏曲知识库文档并入库
python generate_opera_kb.py
```

此脚本会：
1. 批量生成多剧种、多主题的戏曲学术文献（md/txt/pdf）
2. 将文档写入 `knowledge_base/opera/` 目录
3. 逐篇入库至 Chroma 向量库 + BM25 索引

### 运行测试

```bash
# 运行全部测试
python -m pytest tests/ -v

# 运行单个测试文件
python -m pytest tests/test_reranker.py -v
```

## ⚠️ 已知限制

1. **即梦 AI 图像生成**：需要配置火山引擎 AK/SK 并将 `JIMENG_IMAGE_ENABLED` 设为 `True`，否则脸谱画像仅输出文本描述，不生成图片。
2. **模型依赖**：LLM 和嵌入模型均依赖本地 Ollama，需确保 Ollama 服务正常运行且模型已拉取。
3. **Redis 依赖**：缓存和会话记忆功能依赖 Redis，Redis 不可用时会报错，但不会影响核心对话功能（降级处理）。
4. **评测问题**：评测问题集改为动态生成，根据传入主题 + 知识库内容自动生成，不再使用硬编码问题集。
5. **文献生成耗时**：生成较长文献可能需要较长时间，接口超时设为 180 秒，超时返回明确错误。
6. **MCP Server**：为独立进程，需单独启动（`python -m mcp_server.rag_tools_mcp`），不与主服务自动关联。

## 📝 许可证

本项目仅用于学习和研究目的。