# 🎭 RAG知识库智能平台（秋招工程标准版）

基于 **LangChain + LangGraph + Chroma + BM25 + FastAPI + Streamlit** 的一体化RAG平台。

## ✨ 核心功能

| 模块 | 功能说明 |
|------|----------|
| 💬 **智能对话** | 普通单Agent问答 / 多智能体协作 / SSE流式对话 |
| 📚 **知识库管理** | 文档上传入库(txt/md/pdf) / 混合检索测试 / 统计查看 |
| 🎭 **戏曲文献生成** | 一键生成戏曲学术文献，支持 **txt / pdf / md** 三种格式 |
| 📊 **指标评测** | 召回率(Recall@K)、命中率(HitRate@K)、MRR@K、NDCG@K 量化测试 |
| ⚙️ **系统管理** | 健康检查 / 缓存管理 / 日志级别动态调整 |

## 📁 项目结构

```
├── api/                      # FastAPI后端
│   ├── main.py               # 主入口（生命周期/异常/路由注册）
│   ├── schema.py             # Pydantic请求/响应模型
│   ├── limiter.py            # 全局限速器
│   ├── stream_response.py    # 流式响应封装
│   └── routes/               # 业务路由模块
│       ├── chat_routes.py        # 对话接口
│       ├── kb_routes.py          # 知识库管理接口
│       ├── literature_routes.py  # 戏曲文献生成接口
│       ├── evaluation_routes.py  # 检索指标评测接口
│       └── system_routes.py      # 系统管理接口
│
├── agent/                    # Agent智能体
│   ├── graph_base.py         # 单Agent（工具规划-执行-反思）
│   ├── multi_agent.py        # 多Agent（主管-工人模式）
│   ├── session_memory.py     # Redis会话记忆
│   └── memory/               # 🧠 分层隔离记忆模块
│       ├── __init__.py       # 模块入口
│       ├── permanent_memory.py   # 永久静态记忆（系统规则/领域规范/用户偏好）
│       ├── task_memory.py    # 任务级记忆（任务隔离/父子嵌套/中间产物）
│       ├── session_memory.py # 会话时序记忆（对话轨迹/工具调用/自动归档）
│       └── memory_manager.py # 统一记忆管理器（三层协调）
│
├── rag/                      # RAG核心
│   ├── loader/               # 文档加载器(txt/md/pdf)
│   ├── splitter/             # 文本分块器
│   ├── chain/                # RAG链
│   └── vectorstore/          # Chroma向量库 + BM25 + 混合检索
│
├── literature/               # 🎭 戏曲文献生成模块
│   └── generator.py          # 文献生成器(txt/pdf/md输出)
│
├── evaluation/               # 📊 检索指标评测模块
│   └── metrics_evaluator.py  # Recall/HitRate/MRR/NDCG计算与报告生成
│
├── frontend/                 # Streamlit前端
│   ├── app.py                # 首页
│   ├── api_client.py         # 后端API客户端
│   └── pages/                # 多页面
│       ├── 1_💬_智能对话.py
│       ├── 2_📚_知识库管理.py
│       ├── 3_🎭_戏曲文献生成.py
│       └── 4_📊_指标评测.py
│
├── utils/                    # 工具库
│   ├── logger.py             # 日志系统(控制台+文件滚动)
│   ├── rag_exceptions.py     # 统一异常体系
│   ├── exception_handler.py  # 全局异常装饰器
│   ├── cache_utils.py        # 双层Redis缓存
│   └── redis_client.py       # Redis客户端
│
├── config.py                 # 全局配置
├── start_backend.py          # 后端启动脚本
├── start_frontend.py         # 前端启动脚本
│
├── literature_output/        # 📁 文献输出根目录（按类型分子文件夹）
│   ├── txt/                  # TXT文献
│   ├── pdf/                  # PDF文献
│   └── md/                   # MD文献
│
  └── evaluation_report/        # 📁 评测报告保存目录
    ├── report.md             # 检索指标评测报告（最新）
    ├── report_before_layered_memory.md   # 改造前基准报告
    ├── report_after_layered_memory.md    # 改造后报告
    └── memory_layered_comparison_report.md # 前后指标对比文档

memory_store/                 # 📁 分层记忆存储目录
├── permanent_memory.json     # 永久静态记忆（全局共享）
├── tasks/                    # 任务级记忆（每个任务独立文件）
└── sessions/                 # 会话时序记忆（按时序记录）
└── archives/                 # 会话归档（自动清理存储）
```

## 🚀 快速启动

### 前置要求

- Python ≥ 3.11
- [Ollama](https://ollama.ai) 已启动，并已拉取模型：
  ```bash
  ollama pull qwen2:7b      # LLM模型
  ollama pull nomic-embed-text  # 嵌入模型
  ```
- Redis 已启动：
  ```bash
  redis-server
  ```

### 1. 安装依赖（使用uv虚拟环境）

```bash
# 创建虚拟环境并安装全部依赖
uv sync
```

### 2. 启动后端服务

```bash
python start_backend.py
```

启动后自动：
- 初始化Redis连接
- 重建BM25索引
- 构建单/多Agent实例
- 创建文献输出目录

访问 API 文档：http://127.0.0.1:8000/docs

### 3. 启动前端

```bash
python start_frontend.py
```

访问地址：http://localhost:8501

## 🔌 API 接口总览

### 对话接口 `/api/chat/*`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat/normal` | 普通Agent对话 |
| POST | `/api/chat/multi_agent` | 多智能体协作对话 |
| POST | `/api/chat/stream` | 流式对话(SSE) |

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
| POST | `/api/literature/generate` | 生成戏曲文献(txt/pdf/md) |
| GET  | `/api/literature/list` | 文献文件列表 |
| GET  | `/api/literature/read` | 读取文献内容 |
| GET  | `/api/literature/download` | 下载文献文件 |

### 评测接口 `/api/evaluation/*`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/evaluation/run` | 执行检索指标评测 |
| GET  | `/api/evaluation/report` | 读取评测报告 |

### 系统接口 `/api/*`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/api/health` | 健康检查 |
| POST | `/api/cache/clear` | 清空缓存 |
| GET  | `/api/cache/status` | 缓存状态 |
| POST | `/api/log/level` | 日志级别调整 |

## 🎭 戏曲文献生成

支持三种输出格式：
- **TXT**：纯文本格式
- **PDF**：带标题和元数据的PDF文档（自动注册中文字体）
- **MD**：Markdown格式

文献统一存放至 `literature_output/` 根目录，按类型自动分入子文件夹：

```
literature_output/
├── txt/     # .txt 文献
├── pdf/     # .pdf 文献
└── md/      # .md 文献
```

**调用示例：**
```bash
curl -X POST http://127.0.0.1:8000/api/literature/generate \
  -H "Content-Type: application/json" \
  -d '{
    "genre": "京剧",
    "theme": "京剧脸谱艺术研究",
    "length": 1000,
    "formats": ["txt", "pdf", "md"],
    "title": "京剧脸谱研究"
  }'
```

## 📊 检索指标评测

评测指标：
- **Recall@K**：召回率，前K条结果中相关文档占比
- **HitRate@K**：命中率，前K条结果是否至少包含一条相关文档
- **MRR@K**：平均倒数排名，第一个相关文档的位置
- **NDCG@K**：归一化折损累计增益

评测流程：
1. 读取 `config.py` 中的 `EVAL_QUESTION_SET` 测试问题集
2. 对每个问题执行混合检索
3. 按期望关键词判定检索结果相关性
4. 计算各K值下的指标
5. 多次运行取均值
6. 生成Markdown报告保存至 `evaluation_report/report.md`

**调用示例：**
```bash
curl -X POST http://127.0.0.1:8000/api/evaluation/run \
  -H "Content-Type: application/json" \
  -d '{"rounds": 3}'
```

## 🧠 分层隔离记忆系统

Agent长时间记忆采用**分层隔离存储**，禁止全部存入单一向量库：

### 1. 永久静态记忆 `memory_store/permanent_memory.json`

| 类别 | 说明 |
|------|------|
| `system_rules` | 系统规则（检索规范等） |
| `domain_spec` | 领域规范（RAG/LangGraph等） |
| `tech_docs` | 技术文档（Chroma检索等） |
| `user_prefs` | 用户固定偏好 |

- **全局共享**：所有会话/任务通用
- **极少修改**：系统内置 + 手动增删改

### 2. 任务级记忆 `memory_store/tasks/{task_id}.json`

- 每个任务创建独立记忆空间，保存**任务需求/目标/中间产物/约束**
- 任务数据互相隔离，互不干扰
- 支持**父子任务嵌套**（`parent_task_id` / `sub_task_ids`）
- 支持任务的创建、更新、删除（可级联）、完成标记

### 3. 会话时序记忆 `memory_store/sessions/{session_id}.json`

- 记录**对话轨迹、思考、工具调用、结果、报错、修改记录**
- 每条记录附带**时间戳**
- 超过阈值自动**归档**至 `memory_store/archives/`
- 超过保留天数自动**清理**过期日志

### 统一记忆管理器

- `agent/memory/memory_manager.py` 提供统一接口
- Agent各节点自动检索三层记忆并注入提示词
- 原有API接口、参数完全不变，向后兼容

### 统一认知记忆抽象（Working/Episodic/Semantic）🆕

`agent/memory/unified_memory.py` 将三层 JSON Memory 抽象为认知科学标准的三类记忆：

| 认知记忆 | 物理存储 | 说明 |
|---------|---------|------|
| **Working Memory**（工作记忆） | 会话时序记忆 | 短期当前上下文/对话轨迹 |
| **Episodic Memory**（情景记忆） | 任务级记忆 | 任务经历/中间产物 |
| **Semantic Memory**（语义记忆） | 永久静态记忆 | 知识/规则/偏好 |

- 提供 `retrieve_all()` / `format_context()` 统一接口
- 完全兼容旧接口，底层存储不变
- 供 Agent、MCP、Evaluation 统一调用

## 🚀 增量升级特性（保持原有功能完全兼容）

### 1. Hybrid Search + Reranker 精排 🆕

检索流水线升级为：**BM25 + 向量召回 → 分数融合 → LLM Reranker 精排**

- `rag/reranker/llm_reranker.py`：基于 LLM 的二次精排器（0-10 相关性评分）
- 精排失败自动降级为关键词精排，不阻断主流程
- 可通过 `config.ENABLE_RERANKER` 开关控制

### 2. Agentic RAG 🆕

`rag/agentic/` 提供完整 Agentic RAG 组件：

| 组件 | 文件 | 功能 |
|------|------|------|
| QueryRewriter | `query_rewriter.py` | 多轮对话查询改写为独立检索查询 |
| DocumentGrader | `document_grader.py` | 检索文档相关性评分/过滤 |
| AnswerGrader | `answer_grader.py` | 回答忠实度/相关性校验 |
| AutoRetriever | `auto_retriever.py` | 文档不足时自动改写查询重试 |

### 3. MCP Server 🆕

`mcp_server/rag_tools_mcp.py` 将核心能力封装为 4 个 MCP Tools：

| 工具 | 功能 |
|------|------|
| `kb_search` | 混合检索知识库（含精排） |
| `doc_get` | 获取知识库/文献文件内容 |
| `lit_generate` | 生成戏曲文献（txt/pdf/md） |
| `eval_rag` | 执行 RAG 检索指标评测 |

运行方式：`python -m mcp_server.rag_tools_mcp`（JSON-RPC over stdio）

### 4. 高级评测指标 🆕

`evaluation/advanced_evaluator.py` 在基础检索指标上新增：

| 指标 | 说明 |
|------|------|
| Faithfulness | 答案忠实度（无幻觉） |
| Answer Relevance | 答案是否直接回答用户问题 |
| Context Precision | 相关文档排序靠前程度 |
| Context Recall | 相关文档是否都被召回 |
| Agent Task Success | Agent 端到端任务成功率 |

### 5. Agent Trace/Observability 🆕

`agent/trace.py` 提供 Agent 调用可观测性：

- 自动记录每次 Agent 调用的步骤、耗时、工具调用、错误
- 支持装饰器 `@agent_tracer.trace()` 快速接入
- 提供 `get_stats()` 汇总统计（调用次数/成功率/工具使用频率）
- Trace 文件保存至 `agent_traces/` 目录，便于审计

## 🛠️ 工程化特性

- ✅ **统一异常体系**：所有业务异常继承 `BaseRAGException`，包含错误码/错误信息/原始异常
- ✅ **全局异常捕获**：FastAPI全局异常处理器 + 装饰器双重保障
- ✅ **结构化日志**：支持控制台+文件滚动输出，级别可动态调整
- ✅ **双层缓存**：检索缓存 + 问答缓存，Redis实现，可开关控制
- ✅ **限流保护**：所有接口IP限流，防滥用
- ✅ **模块化路由**：按业务域拆分路由，代码结构清晰
- ✅ **安全校验**：文献文件访问路径白名单校验，防目录穿越
- ✅ **分层隔离记忆**：永久静态/任务级/会话时序三层完全隔离
- ✅ **注释完善**：所有模块/函数/方法均有清晰注释

## 📝 配置说明

所有配置集中在 `config.py`：

| 配置项 | 说明 |
|--------|------|
| `LLM_MODEL` | 对话模型名称（Ollama） |
| `EMBED_MODEL` | 嵌入模型名称（Ollama） |
| `CHROMA_PERSIST_PATH` | 向量库持久化路径 |
| `ENABLE_HYBRID_SEARCH` | 是否启用混合检索 |
| `LITERATURE_ROOT_DIR` | 文献输出根目录 |
| `EVAL_QUESTION_SET` | 评测测试问题集 |
| `EVAL_REPORT_PATH` | 评测报告保存路径 |
| `LOG_LEVEL` / `LOG_SAVE_PATH` | 日志级别与保存路径 |