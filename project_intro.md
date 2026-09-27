# Agent-RAG 戏曲科普平台 — 项目介绍文档

> 版本：v2.0.0 | 最后更新：2026-09-07

---

## 1 项目概述

### 1.1 项目背景

本项目是一个面向**中国戏曲文化科普**的 RAG（检索增强生成）智能平台，旨在利用大语言模型（LLM）与本地知识库，为普通用户提供戏曲知识问答、戏词解读、角色对谈、知识闯关、学戏路线规划、脸谱画像生成等一站式戏曲学习与互动体验。

### 1.2 整体技术栈

| 层级 | 技术选型 | 用途 |
|------|----------|------|
| 大模型 | Ollama + qwen2.5:3b | 本地 LLM 推理（对话/生成/意图解析） |
| 嵌入模型 | Ollama + bge-m3 | 文本向量化（中文检索优化） |
| Agent 框架 | LangGraph | 单 Agent ReAct 循环 + 多 Agent 主管-工人协作 |
| 向量库 | Chroma | 文档向量持久化存储与语义检索 |
| 关键词检索 | BM25（rank-bm25） | 关键词稀疏检索，与向量检索混合 |
| 精排 | LLM Reranker | 混合检索结果重排序 |
| 后端 | FastAPI + uvicorn | RESTful API 服务 |
| 前端 | Streamlit | 多页面交互式 Web 界面 |
| 缓存 | Redis | 双层缓存（检索缓存 + 问答缓存）+ 会话存储 |
| 图像生成 | 火山引擎即梦AI（可选） | 脸谱文生图 |
| 文献输出 | Python fpdf / markdown | 戏曲文献生成（txt/pdf/md） |
| 日志 | 自研滚动日志 | 控制台 + 文件滚动输出 |
| 限流 | slowapi | 全局限速保护 |

### 1.3 核心能力一览

1. **RAG 知识库**：覆盖京剧、豫剧、越剧、黄梅戏、昆曲、川剧、评剧、秦腔、粤剧、河北梆子等 10+ 剧种，含 30+ 篇学术级戏曲文献，支持 BM25 + 向量混合检索 + LLM 精排。
2. **三层记忆架构**：永久静态记忆（全局规则/领域知识）、任务级情景记忆（任务隔离）、会话工作记忆（对话轨迹），认知科学视角统一管理。
3. **多 Agent 分工调度**：主管-6 角色化 Worker 模式（search / lyrics / character / quiz / guide / face），支持复合需求串行调度。
4. **动态测评系统**：Recall@K / HitRate@K / MRR@K / NDCG@K 四指标，动态生成测评问题集，多轮取均值。
5. **图片生成（可选）**：脸谱画像模块集成火山引擎即梦 AI 文生图，未配置时自动降级为纯文本版。
6. **Streamlit 前端**：9 页多页面交互，含智能对话、知识库管理、文献生成、指标评测、戏词解剖室、戏中人对谈、知识闯关、学戏路线、脸谱画像。
7. **FastAPI 后端**：模块化路由（6 大业务域），全局异常处理，统一响应格式，SSE 流式对话。

### 1.4 三层记忆说明

| 记忆层级 | 认知科学名称 | 存储位置 | 生命周期 | 说明 |
|----------|-------------|----------|----------|------|
| 永久静态记忆 | 语义记忆（Semantic） | `memory_store/permanent_memory.json` | 全局永久 | 系统规则、戏曲领域规范、用户偏好，所有 Agent 共享 |
| 任务级情景记忆 | 情景记忆（Episodic） | `memory_store/tasks/{task_id}.json` | 任务周期 | 每个任务独立空间，支持父子任务嵌套，记录中间结果与目标 |
| 会话工作记忆 | 工作记忆（Working） | `memory_store/sessions/{session_id}.json` | 会话周期 | 对话轨迹、工具调用记录、思考过程，支持自动归档与过期清理 |

> 三层记忆通过 `MemoryManager` 统一协调，对外提供 `retrieve_for_agent()` 统一检索接口和 `format_memory_context()` 生成 LLM 可读上下文。同时也提供了 `UnifiedCognitiveMemory` 认知科学视角的抽象封装。

---

## 2 项目目录树

```
agent-study-rag/
│
├── api/                                 # 🔌 FastAPI 后端
│   ├── __init__.py
│   ├── main.py                          # 应用入口（生命周期/全局异常/限流/路由注册/静态文件挂载）
│   ├── schema.py                        # Pydantic 请求/响应模型
│   ├── limiter.py                       # 全局限速器（slowapi）
│   ├── stream_response.py               # SSE 流式响应封装
│   └── routes/                          # 业务路由模块
│       ├── chat_routes.py               # 对话接口（普通/多Agent/流式）
│       ├── kb_routes.py                 # 知识库管理接口（入库/检索/统计/清空）
│       ├── literature_routes.py         # 戏曲文献生成接口
│       ├── evaluation_routes.py         # 检索指标评测接口
│       ├── opera_routes.py              # 戏曲科普五大功能接口
│       └── system_routes.py             # 系统管理接口（健康/缓存/日志）
│
├── agent/                               # 🧠 Agent 智能体
│   ├── __init__.py
│   ├── graph_base.py                    # 单 Agent（规划-执行-反思-生成 循环）
│   ├── multi_agent.py                   # 多 Agent（主管-6 角色化 Worker 协作）
│   ├── intent.py                        # 意图解析器（parse_intent 核心）
│   ├── intent_guard.py                  # 意图守卫（安全校验/查询过滤）
│   ├── llm_utils.py                     # LLM 调用工具（重试/超时/安全调用）
│   ├── output_validator.py              # 输出校验器
│   ├── task_context.py                  # 任务上下文管理（隔离记忆检索）
│   ├── trace.py                         # Agent 调用可观测性（Trace 追踪）
│   ├── session_memory.py                # Redis 会话记忆（已废弃/兼容）
│   └── memory/                          # 🧩 分层隔离记忆模块
│       ├── permanent_memory.py          # 永久静态记忆（JSON 文件存储）
│       ├── task_memory.py               # 任务级记忆（按 task_id 隔离）
│       ├── session_memory.py            # 会话时序记忆（事件记录/归档/清理）
│       ├── unified_memory.py            # 统一认知记忆抽象（Working/Episodic/Semantic）
│       └── memory_manager.py            # 统一记忆管理器（协调三层记忆）
│
├── opera/                               # 🎭 戏曲科普业务模块
│   ├── __init__.py                      # 统一导出（五大功能接口）
│   ├── lyrics.py                        # 戏词解剖室（逐句译注/典故/唱腔/心境）
│   ├── character.py                     # 戏中人对谈（角色人格化对话）
│   ├── quiz.py                          # 知识闯关（游戏化出题/判题/讲解）
│   ├── guide.py                         # 个性化学戏路线（阶梯课程/进度追踪）
│   └── face.py                          # 脸谱画像（LLM 设计 + 即梦 AI 文生图）
│
├── rag/                                 # 🔍 RAG 检索增强生成核心
│   ├── __init__.py
│   ├── loader/                          # 文档加载器（txt/md/pdf）
│   ├── splitter/                        # 文本分块器
│   ├── chain/                           # RAG 链
│   ├── reranker/                        # LLM Reranker 精排
│   ├── agentic/                         # Agentic RAG（查询改写/文档评分/自动检索）
│   └── vectorstore/                     # Chroma 向量库 + BM25 索引 + 混合检索
│
├── literature/                          # 📄 戏曲文献生成模块
│   ├── __init__.py
│   └── generator.py                     # 文献生成器（txt/pdf/md 输出）
│
├── evaluation/                          # 📊 检索指标评测模块
│   ├── __init__.py
│   ├── metrics_evaluator.py             # 基础指标（Recall/HitRate/MRR/NDCG）
│   ├── advanced_evaluator.py            # 高级指标（Faithfulness/AnswerRelevance）
│   ├── dynamic_questions.py             # 动态问题生成器
│   ├── eval_agent.py                    # 评测 Agent
│   ├── eval_prompts.py                  # 评测提示词模板
│   └── pre_filter.py                    # 评测前置过滤器
│
├── mcp_server/                          # 🔧 MCP Server
│   ├── __init__.py
│   └── rag_tools_mcp.py                 # JSON-RPC 工具服务（kb_search/doc_get/lit_generate/eval_rag）
│
├── tools/                               # 🛠️ Agent 工具注册表
│   ├── __init__.py
│   └── custom_tools.py                  # 知识库混合检索工具（search_knowledge_base）
│
├── frontend/                            # 🖥️ Streamlit 前端
│   ├── __init__.py
│   ├── app.py                           # 首页（核心功能卡片 + 知识库概览）
│   ├── api_client.py                    # 后端 API 统一客户端（封装所有 HTTP 调用）
│   └── pages/                           # 9 个功能页面
│       ├── 1_💬_智能对话.py              # 智能对话（单Agent / 多Agent 切换）
│       ├── 2_📚_知识库管理.py            # 知识库管理（入库/检索/统计）
│       ├── 3_🎭_戏曲文献生成.py          # 文献生成（流派/主题/格式选择）
│       ├── 4_📊_指标评测.py              # 检索指标评测
│       ├── 5_📜_戏词解剖室.py            # 戏词逐句解读
│       ├── 6_🎭_戏中人对谈.py            # 与戏曲人物对话
│       ├── 7_🎮_知识闯关.py              # 游戏化答题
│       ├── 8_🧭_个性化学戏路线.py        # 阶梯课程规划
│       └── 9_🎨_脸谱画像.py              # 脸谱设计 + 图片生成
│
├── knowledge_base/opera/                # 📚 戏曲知识库文档（多剧种学术文献）
│
├── memory_store/                        # 💾 分层记忆存储目录
│   ├── permanent_memory.json            # 永久静态记忆（全局共享）
│   ├── tasks/                           # 任务级记忆（每个任务独立文件）
│   ├── sessions/                        # 会话时序记忆
│   └── archives/                        # 会话归档（自动清理）
│
├── utils/                               # 🧰 工具库
│   ├── __init__.py
│   ├── logger.py                        # 日志系统（控制台 + 文件滚动）
│   ├── rag_exceptions.py                # 统一异常体系（BaseRAGException）
│   ├── exception_handler.py             # 全局异常装饰器
│   ├── cache_utils.py                   # 双层 Redis 缓存
│   ├── redis_client.py                  # Redis 客户端
│   └── json_repair.py                   # JSON 修复工具（robust_json_loads）
│
├── tests/                               # 🧪 测试套件
│   ├── test_reranker.py
│   ├── test_agentic.py
│   ├── test_trace.py
│   ├── test_mcp.py
│   ├── test_compound_intent.py
│   ├── test_image_scheduling.py
│   ├── test_new_bugs.py
│   └── test_three_fixes.py
│
├── config.py                            # ⚙️ 全局配置文件（所有可调参数）
├── pyproject.toml                       # 项目元数据与依赖声明
├── requirements.txt                     # pip 依赖列表
├── start_backend.py                     # 后端启动脚本
├── start_frontend.py                    # 前端启动脚本
├── generate_opera_kb.py                 # 🔧 工具脚本：批量生成戏曲知识库文档并入库
├── ARCHITECTURE.md                      # 系统架构说明文档
├── README.md                            # 项目说明
└── uv.lock                              # uv 依赖锁定文件
```

> 图例：🔌 后端服务 | 🧠 Agent 调度 | 🎭 戏曲业务 | 🔍 RAG 检索 | 📄 文献生成 | 📊 评测 | 🖥️ 前端 | 📚 知识库 | 💾 记忆存储 | 🧰 工具库 | ⚙️ 配置 | 🧪 测试

---

## 3 核心文件详解

### 3.1 config.py（项目根目录）

> 文件整体作用：全局配置中心，所有可调参数集中管理，无需额外 `.env` 文件。涵盖模型、检索、缓存、日志、文献生成、评测等所有模块的配置项。

关键变量/配置段说明：

- **LLM_MODEL / EMBED_MODEL / LLM_TEMP / LLM_TIMEOUT**：指定 Ollama 本地模型名称、生成温度、调用超时。`LLM_MODEL="qwen2.5:3b"` 是对话模型，`EMBED_MODEL="bge-m3"` 是中文嵌入模型。
- **CHUNK_SIZE=120 / CHUNK_OVERLAP=15**：文档分块大小与重叠量，控制向量化粒度和检索精度。
- **CHROMA_PERSIST_PATH / RETRIEVE_TOP_K / SIMILARITY_THRESHOLD**：向量库持久化路径、默认检索返回条数、相似度过滤阈值（低于 0.6 丢弃）。
- **ENABLE_HYBRID_SEARCH / BM25_TOP_K / VECTOR_TOP_K / HYBRID_FINAL_K**：混合检索开关及参数。BM25+向量各自召回 7 条，融合后最终返回 10 条，权重 VECTOR_WEIGHT=0.6 / BM25_WEIGHT=0.4。
- **ENABLE_RERANKER / RERANKER_TOP_K**：是否启用 LLM 精排，精排后返回 3 条。
- **ENABLE_RAG_CACHE / RETRIEVE_CACHE_TTL / CHAT_CACHE_TTL**：Redis 双层缓存开关及过期时间（检索缓存 100s，问答缓存 100s）。
- **MAX_REFLECT_TIMES=2**：单 Agent 反思重试最大次数。
- **JIMENG_IMAGE_ENABLED / JIMENG_ACCESS_KEY_ID / JIMENG_SECRET_ACCESS_KEY**：即梦 AI 图像生成配置，`JIMENG_IMAGE_ENABLED=False` 默认关闭，开启需配置火山引擎 AK/SK。
- **LOG_LEVEL / LOG_TO_FILE / LOG_SAVE_PATH**：日志级别、是否写入文件、存储路径。
- **EVAL_DYNAMIC_QUESTION_COUNT / EVAL_DYNAMIC_REF_DOCS_K**：动态评测问题生成数量（默认 6 道）和参考文档数。
- **LITERATURE_ROOT_DIR / LITERATURE_DEFAULT_GENRE / LITERATURE_DEFAULT_LENGTH**：文献输出目录、默认剧种（京剧）、默认篇幅（800 字）。

---

### 3.2 agent/memory/memory_manager.py（统一记忆管理器）

> 文件整体作用：**三层记忆的统一协调中枢**。向上层 Agent 提供 `retrieve_for_agent()` 统一检索接口和 `format_memory_context()` 生成 LLM 可读记忆上下文，提供 `record_*` 系列方法记录对话/思考/工具调用/错误/修改等事件。底层委托给 `PermanentMemory`、`TaskMemory`、`SessionTemporalMemory` 三个独立模块，保持各层完全隔离。

关键函数/代码片段解释：

- **class MemoryManager**：统一记忆管理器主类。
  - `__init__(permanent, task, session)`：注入三层记忆实例，默认使用全局单例。
  - `retrieve_for_agent(query, session_id, task_id, include_permanent, include_task, include_session, top_k)`：
    - **作用**：统一检索三层记忆，按需包含各层结果。
    - **入参**：`query` 检索关键词，`session_id` 会话 ID，`task_id` 任务 ID（可选），三个 `include_*` 布尔开关控制检索哪些层，`top_k` 返回条数。
    - **返回值**：`{"permanent": [...], "task": [...], "session": [...]}` 三层结果字典。
    - **业务逻辑**：永久记忆按关键词搜索；任务记忆若指定 task_id 则直接获取该任务摘要（含目标/约束/中间结果），否则按关键词搜索全部任务；会话记忆按关键词搜索指定会话的事件记录。
    - **所处链路**：Agent 在规划/生成阶段调用，获取记忆上下文辅助决策。
  - `format_memory_context(query, session_id, task_id)`：
    - **作用**：将三层记忆检索结果拼接为 LLM 可读的格式化文本。
    - **返回值**：包含 `【永久静态记忆】`、`【任务级记忆】`、`【会话时序记忆】` 三个段落的字符串。
    - **所处链路**：Agent 生成最终回答时，将记忆上下文注入提示词。
  - `record_dialogue(session_id, role, content)` / `record_thinking()` / `record_tool_call()` / `record_tool_result()` / `record_error()` / `record_modify()`：
    - **作用**：记录各类事件到会话时序记忆，供未来检索回溯。
    - **所处链路**：Agent 各节点（规划/执行/反思/生成）在关键节点调用，实时记录运行轨迹。
  - `get_stats()`：返回三层记忆的统计信息（永久记忆条数/分类、任务总数、活跃会话数/归档数）。

- **memory_manager（全局单例）**：`MemoryManager()` 实例，供所有 Agent 节点直接引用。

---

### 3.3 agent/memory/unified_memory.py（统一认知记忆抽象）

> 文件整体作用：将三层 JSON Memory 映射为认知科学标准的三类记忆——**Working Memory（工作记忆）**、**Episodic Memory（情景记忆）**、**Semantic Memory（语义记忆）**。提供 `UnifiedCognitiveMemory` 统一管理器，对外暴露 `retrieve_all()` 和 `format_context()` 接口，同时保持旧接口完全兼容。

关键函数/代码片段解释：

- **class WorkingMemory**：工作记忆，映射到 `SessionTemporalMemory`（会话时序记忆）。
  - `add(session_id, event_type, content, metadata)`：追加一条事件到工作记忆。
  - `get(session_id, limit)`：获取最近 N 条事件。
  - `clear(session_id)`：清理指定会话（归档）。
  - `summarize(session_id)`：获取会话概要。

- **class EpisodicMemory**：情景记忆，映射到 `TaskMemory`（任务级记忆）。
  - `remember(task_id, **kwargs)`：记录一个任务经历（写入 intermediate_results）。
  - `recall(query, top_k)`：按关键词回忆相关任务。
  - `create_episode(**kwargs)`：创建新任务经历空间。

- **class SemanticMemory**：语义记忆，映射到 `PermanentMemory`（永久静态记忆）。
  - `declare(content, category, **kwargs)`：声明一条语义知识。
  - `query(query, top_k)`：语义检索。
  - `forget(mem_id)`：遗忘一条语义知识。

- **class UnifiedCognitiveMemory**：统一认知记忆管理器。
  - `retrieve_all(query, session_id, task_id, top_k)`：统一检索三类记忆，返回 `{"semantic": [...], "episodic": [...], "working": [...]}`。
  - `format_context(query, session_id, task_id)`：生成 LLM 可读的分层记忆上下文，格式为 `【语义记忆（知识/规则）】`、`【情景记忆（任务经历）】`、`【工作记忆（对话上下文）】`。

- **unified_memory（全局单例）**：`UnifiedCognitiveMemory()` 实例。

---

### 3.4 agent/intent.py（意图解析器）

> 文件整体作用：**所有用户请求的入口关卡**——在任何 Agent 处理之前执行意图解析，输出固定 JSON 结构。核心产物：`core_task`（核心任务）、`must_require`（硬性要求）、`forbid_list`（禁止行为）、`is_compound`（是否复合需求）、`sub_tasks`（子任务列表）。解析失败时降级为默认结构，保证下游不阻塞。

关键函数/代码片段解释：

- **WORK_ITEM_DESC**：平台可执行业务功能描述常量，列出 8 种能力（search / lyrics / character / quiz / guide / face / literature / chat），供 LLM 意图解析提示词使用。

- **parse_intent(user_query: str) -> Dict[str, Any]**：
  - **作用**：解析用户原始提问，返回结构化意图。
  - **入参**：`user_query` 用户原始提问文本。
  - **返回值**：`{"core_task": str, "must_require": [str], "optional_require": [str], "forbid_list": [str], "reference_material_required": bool, "is_compound": bool, "sub_tasks": [dict], "raw_query": str}`。
  - **业务逻辑**：
    1. 构建包含平台功能描述和用户提问的提示词，调用 LLM（temperature=0.0）进行解析。
    2. 使用 `robust_json_loads` 容错解析 LLM 输出的 JSON。
    3. 提取 `is_compound` 和 `sub_tasks` 字段支持复合需求识别（如"介绍京剧 + 以杜丽娘口吻回答"）。
    4. 解析失败时降级为 `core_task=原始提问`，所有字段为空/默认值。
  - **所处链路**：前端 → API 路由（chat_routes.py）→ `parse_intent()` → 将结果注入 Agent State 的 `intent` 字段 → 单 Agent / 多 Agent 执行。

- **align_prompt_with_intent(intent: Dict) -> str**：
  - **作用**：将意图解析结果转换为面向下游 Agent/生成模块的"意图对齐提示词片段"。
  - **返回值**：包含 `【当前唯一核心任务】`、`【必须遵守的硬性要求】`、`【禁止行为】`、`【文档约束】`、`【优先级】` 等段落的提示词文本。
  - **所处链路**：Agent 的规划节点和生成回答节点将其拼入最终提示词，确保生成严格对齐用户意图。

- **is_forbidden(query, forbid_list)**：校验用户请求是否踩中 forbid_list。当前为占位实现，真正约束由 LLM 提示词完成。

---

### 3.5 agent/multi_agent.py（多智能体协作）

> 文件整体作用：**主管-工人（Supervisor-Worker）多 Agent 协作引擎**。基于 LangGraph 构建状态图，主管识别意图后分派给 6 个角色化 Worker 执行，支持复合需求多 Worker 串行调度、检索反思重试、图片资源收集与合并。最终由汇总节点整合所有 Worker 产出生成回答。

关键函数/代码片段解释：

- **class MultiAgentState(TypedDict)**：多 Agent 状态定义。
  - `user_query`：用户原始提问。
  - `sub_task_list`：主管分派的任务列表。
  - `worker_result`：各 Worker 执行结果累积。
  - `retry_times` / `need_retry`：检索重试控制。
  - `intent`：意图解析结果。
  - `blackboard`：黑板（Worker 间共享数据）。
  - `image_resources`：图片资源列表（face_worker 产出的真实图片路径）。
  - `accumulated_worker_result`：多轮检索结果累积。
  - `_task_index` / `_compound_workers`：复合需求串行调度控制。

- **supervisor_node(state) -> MultiAgentState**：
  - **作用**：主管节点，识别用户意图并分派任务给最匹配的 Worker。
  - **业务逻辑**：
    1. 注入意图约束（`core_task` 最高优先级、`forbid_list` 绝对不违反）。
    2. 注入检索反思反馈（上一轮检索不足时，提供差异化的重规划建议）。
    3. 调用 LLM 判断任务类型，输出 `{"worker": "xxx", "task": "...", "params": {...}}`。
    4. 使用 `intent_guard.filter_tasks_by_intent` 进行意图校验，过滤不允许的 Worker。
  - **所处链路**：多 Agent 图的入口节点，所有请求必经此节点。

- **search_worker(state) -> MultiAgentState**：
  - **作用**：知识检索工人，调用 `hybrid_retrieve` 执行混合检索。
  - **业务逻辑**：重试时跳过缓存（`skip_cache=True`），结果不足时触发重试（`need_retry=True`），多轮结果累积（`accumulated_worker_result`），记录 `last_search_query` 供下一轮反思禁止重复。
  - **所处链路**：主管分派 → search_worker → 反思判断 → 重试或汇总。

- **lyrics_worker(state)**：戏词解剖工人，调用 `opera.annotate_lyrics`。
- **character_worker(state)**：人物对谈工人，调用 `opera.character_chat`，携带最近 8 条对话历史。
- **quiz_worker(state)**：知识闯关工人，调用 `generate_quiz` 或 `generate_quiz_batch`（动态题量），从用户提问中提取题目数量。
- **guide_worker(state)**：学戏路线工人，调用 `opera.generate_course`。
- **face_worker(state)**：脸谱画像工人，调用 `opera.generate_face_profile`，并收集图片资源到 `image_resources` 列表。

- **summary_agent(state) -> MultiAgentState**：
  - **作用**：汇总节点，整合所有 Worker 结果生成最终回答。
  - **业务逻辑**：
    1. 检测 face_worker 是否为唯一 Worker → 是则结构化直出（保留完整图片 URL）。
    2. 复合需求含 face_worker 时 → 走 LLM 汇总，但强制保留图片资源。
    3. 注入意图对齐片段、隔离记忆上下文、过滤后的历史对话。
    4. 按 Worker 类型定制汇总要求（戏词解剖→结构化输出、人物对谈→角色口吻、知识闯关→题目+选项等）。
  - **所处链路**：所有 Worker 执行完毕 → summary_agent → 返回最终回复。

- **worker_route / search_finish_route / next_worker_route**：路由函数。
  - `worker_route`：按主管分派的任务路由到对应 Worker。
  - `search_finish_route`：检索结果不足→重试（回 supervisor_node），充足→下一 Worker 或汇总。
  - `next_worker_route`：复合需求中非检索 Worker 完成后，路由到下一个 Worker。

- **build_multi_agent()**：
  - **作用**：构建 LangGraph 多 Agent 状态图。
  - **图结构**：`supervisor_node` →（条件路由）→ 6 个 Worker → `compound_worker_finish` →（条件路由）→ 下一个 Worker 或 `summary_agent` → END。
  - **所处链路**：在 `api/main.py` 的 lifespan 启动阶段调用，构建后存入 `app.state.multi_agent`。

---

### 3.6 agent/graph_base.py（单 Agent 图）

> 文件整体作用：**单 Agent ReAct 循环引擎**。基于 LangGraph 构建"规划→执行→反思→生成"四节点状态图，支持工具调用（知识库检索）、反思重试（结构化诊断报告）、意图约束、任务隔离记忆、Trace 可观测性。

关键函数/代码片段解释：

- **class AgentState(TypedDict)**：单 Agent 状态定义。
  - `messages`：对话消息累积。
  - `user_query` / `tool_call` / `tool_result`：用户提问、工具调用信息、工具执行结果。
  - `reflect_times`：当前反思次数。
  - `intent`：意图解析结果。
  - `last_reflection`：上次反思的结构化诊断报告（含 `sufficient`/`missing_aspects`/`suggested_queries`/`reason`）。
  - `last_query`：上次检索 query（禁止重复）。
  - `accumulated_results`：多轮检索结果累积。

- **plan_tool_call(state) -> AgentState**：
  - **作用**：规划节点，判断是否需要调用工具以及调用哪个工具。
  - **业务逻辑**：
    1. 获取任务隔离记忆上下文（`get_current_task_only`）。
    2. 注入意图约束（`core_task` 最高优先级、`forbid_list`）。
    3. 注入上次反思反馈（结构化诊断报告），驱动差异化重规划。
    4. 调用 LLM 输出 `{"tool_name": "search_knowledge_base", "params": {"query": "xxx"}}` 或 `{"tool_name": "none"}`。
    5. 意图校验：日常闲聊关键词（"你好"/"谢谢"等）强制不调用工具。
  - **所处链路**：单 Agent 图的入口节点。

- **run_tool(state) -> AgentState**：
  - **作用**：执行工具节点，调用 `knowledge_tool.invoke(params)`。
  - **业务逻辑**：重试时累积检索结果到 `accumulated_results`，记录 `last_query` 供下一轮反思禁止重复。
  - **所处链路**：plan_tool_call → run_tool → reflect_check。

- **reflect_check(state) -> AgentState**：
  - **作用**：反思校验节点，判断当前检索结果是否足够回答用户问题。
  - **业务逻辑**：调用 LLM 输出结构化诊断报告（`{"sufficient": bool, "missing_aspects": [...], "suggested_queries": [...], "reason": "..."}`）。不足时递增 `reflect_times` 并返回 `last_reflection`，达到上限 `MAX_REFLECT_TIMES` 时强制终止。
  - **所处链路**：run_tool → reflect_check →（不足）→ plan_tool_call（重试）或 → generate_final_ans。

- **generate_final_ans(state) -> AgentState**：
  - **作用**：生成最终回答节点。
  - **业务逻辑**：注入意图对齐片段、隔离记忆上下文、过滤后的历史对话（仅最近 2 条用户消息摘要），调用 LLM 生成回复。
  - **所处链路**：reflect_check（充足）→ generate_final_ans → END。

- **route_by_tool / route_reflect**：路由函数。
  - `route_by_tool`：有工具调用 → run_tool，无工具调用 → generate_final_ans。
  - `route_reflect`：根据诊断报告的 `sufficient` 字段决定重试或生成。

- **build_agent_graph()**：
  - **作用**：构建 LangGraph 单 Agent 状态图，并包装 `invoke` 方法自动开启/结束 Trace。
  - **图结构**：`plan_tool_call` →（条件路由）→ `run_tool` → `reflect_check` →（条件路由）→ `plan_tool_call`（重试）或 `generate_final_ans` → END。
  - **所处链路**：在 `api/main.py` 的 lifespan 启动阶段调用，构建后存入 `app.state.single_agent`。

---

### 3.7 generate_opera_kb.py（知识库批量生成脚本）

> 文件整体作用：**一次性工具脚本**，批量生成 30+ 篇涵盖 10 个剧种的戏曲学术文献（md/txt），写入 `knowledge_base/opera/` 目录，并逐篇入库至 Chroma 向量库 + BM25 索引。执行后会彻底清除旧向量库数据并重建。

关键函数/代码片段解释：

- **OPERA_DOCS**：数据集常量，包含 30+ 个元组 `(剧种, 主题, 文件名, 内容)`，覆盖京剧、豫剧、越剧、黄梅戏、昆曲、川剧、评剧、秦腔、粤剧、河北梆子十大剧种。每篇文献约 500-800 字，内容涵盖唱腔、脸谱、表演程式、代表剧目、流派艺术等。

- **reset_chroma_storage()**：删除 `chroma_db` 持久化目录，实现向量库干净重建。

- **generate_all_docs()**：
  - **作用**：主函数，执行完整的知识库重建流程。
  - **业务逻辑**：
    1. 清空内存向量集合 + 删除磁盘持久化目录。
    2. 重建 `ChromaKnowledgeBase` 实例，重置 BM25 索引。
    3. 清空 `knowledge_base/opera/` 旧文档。
    4. 遍历 OPERA_DOCS，逐篇写入文件 → 调用 `kb.add_file_increment()` 入库。
    5. 调用 `kb.rebuild_full_bm25()` 重建 BM25 索引。
    6. 输出统计信息。
  - **使用方式**：`python generate_opera_kb.py`。

---

### 3.8 frontend/api_client.py（前端 API 客户端）

> 文件整体作用：**Streamlit 前端与 FastAPI 后端之间的 HTTP 通信桥梁**。封装所有后端 API 接口为统一的方法调用，处理连接错误、超时、HTTP 错误的统一异常捕获，返回标准化 `{"code": ..., "msg": ..., "data": ...}` 格式。

关键函数/代码片段解释：

- **class ApiClient**：统一 API 客户端类。
  - `__init__(base_url)`：初始化后端地址，默认 `http://127.0.0.1:8000`，默认 `session_id="streamlit_user"`。
  - `_post(path, data)` / `_get(path, params)`：HTTP 基础封装，自动拼接 base_url，统一超时设置（POST 600s / GET 60s），捕获 `ConnectionError`/`Timeout`/`HTTPError` 并返回标准化错误响应。
  - **对话接口**：`chat_normal(query)` → POST `/api/chat/normal`；`chat_multi_agent(query)` → POST `/api/chat/multi_agent`。
  - **知识库接口**：`kb_ingest(file_path)` / `kb_retrieve(query)` / `kb_stats()` / `kb_clear()` / `kb_upload(file_bytes, filename)`。
  - **文献生成接口**：`literature_generate(genre, theme, length, formats, title)` / `literature_list()` / `literature_read(path)` / `literature_download_url(path)` / `literature_status(task_id)`。
  - **评测接口**：`evaluation_run(rounds)` / `evaluation_report()`。
  - **戏曲科普接口**：`lyrics_annotate(lyrics)` / `character_list()` / `character_chat(character_id, message)` / `quiz_generate(topic, difficulty)` / `quiz_topics()` / `quiz_check(...)` / `course_generate(topic, days, level)` / `course_progress()` / `face_generate(preferences)`。
  - **系统管理接口**：`health_check()` / `cache_status()` / `cache_clear()`。
  - **所处链路**：前端 Streamlit 页面 → `api_client.xxx()` → HTTP 请求 → FastAPI 后端路由 → 业务处理 → 返回 JSON。

- **api_client（全局单例）**：`ApiClient()` 实例，供所有 Streamlit 页面直接引用。

---

### 3.9 frontend/pages/1_💬_智能对话.py（智能对话页面）

> 文件整体作用：Streamlit 前端智能对话页面，支持单 Agent 和多 Agent 两种模式切换，维护对话历史，展示文本回复和图片资源（脸谱生成结果）。

关键函数/代码片段解释：

- **页面结构**：
  - `mode` 单选按钮：切换"普通智能体（单Agent）"和"多智能体协作"。
  - `query` 文本输入框：用户输入问题。
  - 发送按钮触发 `api_client.chat_normal(query)` 或 `api_client.chat_multi_agent(query)`。
  - 对话历史存储在 `st.session_state.chat_history`。
  - 图片资源渲染：检查 `data.images` 列表，使用 `st.image()` 展示真实生成的图片。
  - 图片生成失败标记：`image_generation_failed` 为 True 时显示警告提示。

- **所处链路**：用户输入 → 前端调用 `api_client` → 后端处理 → 返回 JSON → 前端解析展示。

---

### 3.10 tools/custom_tools.py（Agent 工具注册表）

> 文件整体作用：为单 Agent 注册可用工具。当前仅注册 `search_knowledge_base` 一个工具（知识库混合检索），戏曲科普五功能已下沉到 `opera/` 业务模块，由多 Agent 编排调用。

关键函数/代码片段解释：

- **search_knowledge_base(query: str) -> str**：
  - **作用**：执行知识库混合检索，返回检索结果文本。
  - **入参**：`query` 检索查询文本。
  - **返回值**：检索到的文档片段拼接字符串，格式为 `"文档片段：xxx，来源：xxx"`。
  - **业务逻辑**：调用 `hybrid_retrieve(query)` 执行 BM25+向量混合检索，空结果返回提示文本。
  - **所处链路**：单 Agent → plan_tool_call → run_tool → `knowledge_tool.invoke(params)`。

- **knowledge_tool**：`StructuredTool.from_function(search_knowledge_base)` 包装的 LangChain 工具。
- **tool_list**：`[knowledge_tool]`，供单 Agent 规划节点提示词使用。

---

### 3.11 opera/__init__.py（戏曲科普业务模块入口）

> 文件整体作用：统一导出五大戏曲科普功能的公共接口，供多 Agent Worker 和 API 路由直接引用。

导出清单：

| 导出名称 | 来源文件 | 功能 |
|----------|----------|------|
| `annotate_lyrics` | `opera.lyrics` | 戏词解剖（逐句翻译/典故/心境/唱腔） |
| `get_character_profile` | `opera.character` | 获取角色档案 |
| `character_chat` | `opera.character` | 与角色人格化对话 |
| `CHARACTER_LIST` | `opera.character` | 内置角色列表 |
| `generate_quiz` | `opera.quiz` | 单题生成 |
| `generate_quiz_batch` | `opera.quiz` | 批量出题 |
| `check_answer` | `opera.quiz` | 判题+讲解 |
| `QUIZ_TOPICS` | `opera.quiz` | 可用出题主题 |
| `generate_course` | `opera.guide` | 生成学戏路线 |
| `get_course_progress` | `opera.guide` | 查询学戏进度 |
| `generate_face_profile` | `opera.face` | 生成脸谱画像 |

---

### 3.12 api/main.py（FastAPI 后端入口）

> 文件整体作用：FastAPI 应用主入口，负责应用生命周期管理（启动时初始化 Redis/BM25/Agent/文献目录/记忆模块，关闭时清理 Redis）、全局异常处理（业务异常 + 兜底未知异常）、限流绑定、路由注册、静态文件挂载。

关键函数/代码片段解释：

- **lifespan(app: FastAPI)**：异步上下文管理器，管理启动/关闭生命周期。
  - **启动阶段**：初始化 Redis → 重建 BM25 索引 → 构建单 Agent 和多 Agent 实例并存入 `app.state` → 确认文献输出目录 → 初始化分层记忆模块（清理过期会话）。
  - **关闭阶段**：关闭 Redis 连接。

- **app = FastAPI(...)**：应用实例，title="本地Agent知识库后端"，version="2.0.0"。

- **rag_exception_handler / global_unknown_exception_handler**：全局异常处理器。
  - `BaseRAGException` 子类异常 → 返回 `{"code": exc.code, "msg": exc.msg, "detail": "..."}`。
  - 未知异常 → 返回 `{"code": 9999, "msg": "服务内部未知错误", "detail": "..."}`。

- **路由注册**：6 个业务路由模块挂载。
  - `chat_router` → `/api/chat/*`
  - `kb_router` → `/api/kb/*`
  - `literature_router` → `/api/literature/*`
  - `evaluation_router` → `/api/evaluation/*`
  - `system_router` → `/api/health` `/api/cache/*` `/api/log/*`
  - `opera_router` → `/api/opera/*`

- **静态文件挂载**：`literature_output/` 目录挂载到 `/static`，供前端访问生成的图片。

---

## 4 整体业务运行流程

以下完整描述用户提问从前端输入到结果返回的全链路：

```
┌─────────────────────────────────────────────────────────────────────┐
│                     用户提问（Streamlit 前端）                       │
│   frontend/pages/1_💬_智能对话.py                                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP POST
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│               api_client.py  →  FastAPI 路由层                       │
│   POST /api/chat/multi_agent 或 /api/chat/normal                     │
│   api/routes/chat_routes.py                                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ① 意图解析 parse_intent()                          │
│   agent/intent.py                                                   │
│   输入：user_query（用户原始提问）                                    │
│   输出：{core_task, must_require, forbid_list, is_compound,          │
│           sub_tasks, reference_material_required}                    │
│   失败降级：core_task=原始提问，所有字段为空                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ intent 注入 Agent State
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              ② 主管 Agent 调度（仅多 Agent 模式）                     │
│   agent/multi_agent.py :: supervisor_node                            │
│   输入：user_query + intent + 反思反馈（如有）                        │
│   输出：{worker: "xxx", task: "xxx", params: {...}}                  │
│   规则：用户当前核心任务 = 最高优先级，forbid_list 绝对不违反         │
│   意图校验：intent_guard.filter_tasks_by_intent 过滤非法 Worker       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ 路由到对应 Worker
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ③ Worker 执行（6 选 1 或多选串行）                   │
│   agent/multi_agent.py                                              │
│                                                                     │
│   ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│   │ search_worker   │  │ lyrics_worker    │  │ character_worker │   │
│   │ → hybrid_retrieve│  │ → annotate_lyrics│  │ → character_chat │   │
│   │ (BM25+向量+精排) │  │ (逐句译注/典故)  │  │ (角色人格化对话) │   │
│   └─────────────────┘  └──────────────────┘  └──────────────────┘   │
│   ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│   │ quiz_worker     │  │ guide_worker     │  │ face_worker      │   │
│   │ → generate_quiz │  │ → generate_course│  │ → generate_face  │   │
│   │ (出题/批量出题) │  │ (阶梯课程)       │  │ (脸谱+即梦AI图)  │   │
│   └─────────────────┘  └──────────────────┘  └──────────────────┘   │
│                                                                     │
│   检索重试：结果为空/过短 → need_retry=True → 回 supervisor_node     │
│   重试上限：MAX_RETRY=2 次                                           │
│   复合需求：search_worker 完成后 → compound_worker_finish            │
│            → next_worker_route → 下一个 Worker                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ worker_result 累积
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              ④ 记忆读写（贯穿全流程）                                 │
│   agent/memory/memory_manager.py                                    │
│   写入：各节点实时记录对话/思考/工具调用/工具结果/错误/修改           │
│   读取：规划/生成阶段调用 retrieve_for_agent / format_memory_context │
│   三层记忆：永久静态（全局规则）→ 任务级（任务隔离）→ 会话（时序轨迹）│
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ⑤ 汇总节点 summary_agent                           │
│   agent/multi_agent.py :: summary_agent                             │
│   输入：所有 worker_result + intent + 记忆上下文 + 历史对话过滤       │
│   处理：                                                            │
│   - face_worker 唯一 Worker → 结构化直出（保留图片 URL）             │
│   - 复合需求含 face_worker → LLM 汇总 + 强制保留图片资源             │
│   - 其他场景 → LLM 汇总（注入意图/记忆/历史约束）                     │
│   输出：最终回复文本 + 图片资源列表                                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              ⑥ 返回前端（JSON 响应）                                 │
│   api/routes/chat_routes.py → FastAPI Response                      │
│   {code: 200, data: {reply: "...", images: [...]}}                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              ⑦ 前端渲染展示                                          │
│   frontend/pages/1_💬_智能对话.py                                    │
│   - 文本回复：st.chat_message + st.markdown                          │
│   - 图片资源：st.image() 展示真实生成的图片                          │
│   - 错误提示：红色错误消息                                           │
└─────────────────────────────────────────────────────────────────────┘
```

> 单 Agent 模式（`/api/chat/normal`）的流程类似，但跳过主管调度，直接走 `plan_tool_call → run_tool → reflect_check → generate_final_ans` 四节点循环。

---

## 5 现存已知问题与待优化点

### 5.1 意图解析字段丢失

- **现象**：`parse_intent()` 返回的 `is_compound` 和 `sub_tasks` 字段在某些 LLM 输出格式不稳定时可能丢失，导致复合需求被当作单一任务处理。
- **原因**：LLM 输出 JSON 格式不稳定，`robust_json_loads` 虽能修复部分格式错误，但对字段缺失无能为力。
- **影响**：用户说"介绍京剧并生成脸谱"时，可能只执行了 search_worker 而跳过了 face_worker。

### 5.2 多 Agent 图片返回丢失

- **现象**：复合需求（如"介绍曹操并生成脸谱"）中 face_worker 产出的图片 URL 在汇总节点可能丢失，前端只显示文字描述。
- **原因**：`summary_agent` 中图片资源兜底收集逻辑依赖于 `face_result` 变量的正确赋值，Worker 结果列表中 `face_worker` 的识别依赖 `w.get("worker") == "face_worker"` 判断，若 Worker 结果结构异常则漏收集。
- **已修复部分**：已在 `summary_agent` 和 `face_worker` 中增加了兜底收集逻辑，但仍需持续验证。

### 5.3 动态测评检索异常

- **现象**：动态测评问题生成时，从知识库检索的参考文档可能与评测主题不相关，导致生成的问题质量低。
- **原因**：`EVAL_DYNAMIC_REF_DOCS_K=5` 条参考文档可能不包含足够的主题覆盖度，且缺少对检索结果相关性的二次校验。
- **影响**：评测指标可能不能真实反映检索系统的能力。

### 5.4 记忆污染

- **现象**：旧任务的工具输出（如上一次的闯关题目、文献生成结果）可能混入当前任务的生成上下文。
- **原因**：三层记忆的隔离依赖 `get_current_task_only()` 正确过滤，但历史对话过滤（`_filter_history_hint`）仅提取最近 2 条用户消息摘要，如果用户在短时间内提出多个不同类型的任务，旧任务的 AI 输出可能被新任务的 LLM 误读。
- **已修复部分**：已在 `generate_final_ans` 和 `summary_agent` 中增加"历史对话参考仅用于语气/人设"的约束，并强制使用 `get_current_task_only` 隔离记忆。

### 5.5 JSON 解析容错

- **现象**：LLM 输出的 JSON 格式不稳定（多余文字、未闭合引号、嵌套错误），导致主管分派和意图解析失败。
- **原因**：`qwen2.5:3b` 模型在低温度下仍可能输出非标准 JSON。
- **已有缓解**：`robust_json_loads`（`utils/json_repair.py`）提供多层修复策略，但无法修复所有格式错误。
- **降级策略**：解析失败时降级为默认值（`search_worker` 或 `core_task=原始提问`）。

### 5.6 检索缓存时效

- **现象**：`RETRIEVE_CACHE_TTL=100` 秒较短，`CHAT_CACHE_TTL=100` 秒同样较短，频繁的相似问题可能无法命中缓存。
- **建议**：根据实际使用场景调整缓存过期时间。

### 5.7 即梦 AI 图像生成依赖

- **现象**：脸谱画像功能需要配置火山引擎 AK/SK 并安装 `volcengine-python-sdk`，未配置时自动降级为纯文本版。
- **影响**：用户可能期望看到真实图片但只得到文字描述。

---

## 6 项目部署运行说明

### 6.1 环境依赖

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Python | ≥ 3.11 | 运行环境 |
| Ollama | 最新版 | 本地 LLM 运行环境 |
| Redis | 最新版 | 缓存与会话存储 |
| uv（推荐） | 最新版 | Python 包管理器 |

### 6.2 安装步骤

```bash
# 1. 克隆项目
git clone <repository-url>
cd agent-study-rag

# 2. 安装依赖（推荐使用 uv）
uv sync

# 或使用 pip
pip install -r requirements.txt

# 3. 启动 Ollama 并拉取模型
ollama pull qwen2.5:3b
ollama pull bge-m3

# 4. 启动 Redis
redis-server    # Linux/macOS
redis-server.exe  # Windows
```

### 6.3 知识库重建

```bash
# 首次部署或需要重建知识库时执行
python generate_opera_kb.py
```

此脚本会：
1. 清除旧 Chroma 向量库持久化数据
2. 批量生成 30+ 篇戏曲学术文献
3. 逐篇入库至 Chroma 向量库 + BM25 索引
4. 重建 BM25 全文索引

### 6.4 启动服务

```bash
# 启动后端（终端1）
python start_backend.py
# 访问 API 文档：http://127.0.0.1:8000/docs

# 启动前端（终端2）
python start_frontend.py
# 访问前端页面：http://localhost:8501
```

后端启动后自动完成：
- Redis 连接初始化
- BM25 关键词索引重建
- 单 Agent / 多 Agent 实例构建
- 文献输出目录创建
- 分层记忆模块初始化（自动清理过期会话）

### 6.5 可选：启用即梦 AI 图像生成

编辑 `config.py`：

```python
JIMENG_IMAGE_ENABLED = True
JIMENG_ACCESS_KEY_ID = "你的火山引擎 AccessKey ID"
JIMENG_SECRET_ACCESS_KEY = "你的火山引擎 Secret Access Key"
```

安装可选依赖：`pip install volcengine-python-sdk`

### 6.6 运行测试

```bash
# 运行全部测试
python -m pytest tests/ -v

# 运行单个测试文件
python -m pytest tests/test_reranker.py -v
```

---

> 文档结束。本文件基于项目源代码真实结构生成，未增删任何代码文件。