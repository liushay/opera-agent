# 🎭 戏曲科普 RAG 知识库平台 — 代码详解

> 项目：agent-study-rag  
> 技术栈：Python 3.11+ / LangChain / LangGraph / FastAPI / Streamlit / ChromaDB / Ollama  
> 本地模型：qwen2.5:3b（生成）、bge-m3（嵌入）、deepseek-r1:7b（评估）

---

## 目录

1. [项目总览与架构](#1-项目总览与架构)
2. [配置层：config.py](#2-配置层configpy)
3. [RAG 检索管线：rag/ 模块](#3-rag-检索管线rag-模块)
4. [Agent 智能体核心：agent/ 模块](#4-agent-智能体核心agent-模块)
5. [Opera 戏曲业务：opera/ 模块](#5-opera-戏曲业务opera-模块)
6. [文献生成：literature/ 模块](#6-文献生成literature-模块)
7. [评估体系：evaluation/ 模块](#7-评估体系evaluation-模块)
8. [API 服务层：api/ 模块](#8-api-服务层api-模块)
9. [前端展示层：frontend/ 模块](#9-前端展示层frontend-模块)
10. [工具与基础设施：tools/、utils/ 模块](#10-工具与基础设施tools-utils-模块)
11. [MCP 服务：mcp_server/ 模块](#11-mcp-服务mcp_server-模块)
12. [记忆系统：agent/memory/ 模块](#12-记忆系统agentmemory-模块)
13. [知识库构建：generate_opera_kb.py](#13-知识库构建generate_opera_kbpy)

---

## 1. 项目总览与架构

### 1.1 项目定位

这是一个**本地化部署的戏曲科普 RAG 知识库平台**。核心能力包括：

- 🔍 **知识检索问答**：混合检索（BM25 + 向量）+ 重排序 + 缓存
- 🤖 **双 Agent 引擎**：单 Agent（规划→执行→反思→生成）+ 多 Agent（主管-工人模式）
- 🎭 **五大戏曲科普功能**：戏词解剖、戏中人对谈、知识闯关、学戏路线、脸谱画像
- 📚 **文献生成**：LLM 生成戏曲文献，支持 txt/pdf/md 三种格式
- 📊 **检索指标评测**：Recall@K / MRR / NDCG，支持动态生成测评问题
- 🧪 **独立评估 Agent**：5 步深度评估流程，幻觉检测 + 意图对齐 + Critic-Refine 循环

### 1.2 架构图

```
┌──────────────────────────────────────────────────────────────┐
│                      Streamlit 前端                          │
│  ┌──────┐ ┌───────┐ ┌───────┐ ┌────────┐ ┌───────┐        │
│  │智能对话│ │戏词解剖│ │戏中人 │ │知识闯关│ │脸谱画像│ ...    │
│  └──┬───┘ └───┬───┘ └───┬───┘ └───┬────┘ └───┬───┘        │
│     │         │         │         │          │              │
│     └─────────┴─────────┴─────────┴──────────┘              │
│                        │ api_client.py                       │
└────────────────────────┼─────────────────────────────────────┘
                         │ HTTP (JSON)
┌────────────────────────┼─────────────────────────────────────┐
│                FastAPI 后端 (api/main.py)                     │
│  ┌─────────────────────┼─────────────────────────────────┐   │
│  │              路由层 (api/routes/)                      │   │
│  │  /chat  /kb  /literature  /evaluation  /opera  /system│   │
│  └─────────────────────┼─────────────────────────────────┘   │
│                        │                                      │
│  ┌─────────────────────┼─────────────────────────────────┐   │
│  │              Agent 引擎                                │   │
│  │  ┌─────────────────┐  ┌──────────────────────────┐   │   │
│  │  │ 单Agent (反思)   │  │ 多Agent (主管-工人)      │   │   │
│  │  │ plan→exec→reflect│  │ supervisor→6 workers     │   │   │
│  │  └────────┬────────┘  └───────────┬──────────────┘   │   │
│  │           │                       │                   │   │
│  │           └───────────┬───────────┘                   │   │
│  │                       │                               │   │
│  │  ┌────────────────────┼──────────────────────────┐    │   │
│  │  │              RAG 检索管线                       │    │   │
│  │  │  缓存→BM25+向量融合→LLM精排(Reranker)           │    │   │
│  │  └────────────────────┼──────────────────────────┘    │   │
│  └───────────────────────┼───────────────────────────────┘   │
│                          │                                    │
│  ┌───────────────────────┼───────────────────────────────┐   │
│  │              数据存储层                                │   │
│  │  ChromaDB(向量)  BM25索引  Redis(缓存/会话)  JSON文件  │   │
│  └───────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### 1.3 目录结构

```
agent-study-rag/
├── config.py              # 全局配置（模型/检索/缓存/文献/评测）
├── start_backend.py       # 后端启动脚本
├── start_frontend.py      # 前端启动脚本
├── generate_opera_kb.py   # 知识库构建脚本
├── agent/                 # 智能体核心
│   ├── graph_base.py      # 单Agent流程图（Plan-Exec-Reflect-Generate）
│   ├── multi_agent.py     # 多Agent协作图（Supervisor-Worker模式）
│   ├── intent.py          # 意图解析层
│   ├── intent_guard.py    # 意图校验守卫
│   ├── output_validator.py# 输出校验层
│   ├── llm_utils.py       # LLM调用工具（重试/超时/日志）
│   ├── task_context.py    # 任务上下文隔离
│   ├── trace.py           # 执行追踪
│   └── memory/            # 三层记忆系统
│       ├── session_memory.py
│       ├── permanent_memory.py
│       └── episodic_memory.py
├── rag/                   # 检索增强生成管线
│   ├── vectorstore/       # 向量库（ChromaDB + BM25）
│   │   ├── chroma_kb.py
│   │   ├── bm25_retriever.py
│   │   └── hybrid_search.py  # 混合检索主入口
│   ├── reranker/          # LLM精排器
│   │   └── llm_reranker.py
│   ├── agentic/           # Agentic RAG
│   │   ├── auto_retriever.py   # 自动检索重试
│   │   ├── query_rewriter.py   # 查询改写
│   │   └── document_grader.py  # 文档评分
│   ├── loader/            # 文档加载
│   └── splitter/          # 文档分块
├── opera/                 # 戏曲业务模块（五大功能）
│   ├── lyrics.py          # 戏词解剖室
│   ├── character.py       # 戏中人对谈
│   ├── quiz.py            # 知识闯关
│   ├── guide.py           # 学戏路线
│   └── face.py            # 脸谱画像（含火山引擎即梦AI图像生成）
├── literature/            # 文献生成
│   └── generator.py       # LLM文献生成(txt/pdf/md)
├── evaluation/            # 评估体系
│   ├── eval_agent.py      # 独立评估Agent（5步流程）
│   ├── eval_prompts.py    # 评估提示词库
│   ├── pre_filter.py      # 快速预筛选
│   ├── metrics_evaluator.py # 检索指标评测
│   └── dynamic_questions.py # 动态测评问题生成
├── api/                   # FastAPI后端服务
│   ├── main.py            # 应用入口（路由/生命周期）
│   ├── schema.py          # 请求/响应模型
│   ├── limiter.py         # 速率限制
│   ├── stream_response.py # SSE流式响应
│   └── routes/            # 路由模块
├── frontend/              # Streamlit前端
│   ├── app.py             # 主入口
│   ├── api_client.py      # 后端API客户端
│   └── pages/             # 各功能页面
├── tools/                 # 工具注册
│   └── custom_tools.py    # 知识库检索工具
├── utils/                 # 基础设施
│   ├── logger.py          # 日志系统
│   ├── cache_utils.py     # Redis缓存
│   ├── redis_client.py    # Redis客户端
│   ├── json_repair.py     # 容错JSON解析
│   └── rag_exceptions.py  # 异常体系
├── mcp_server/            # MCP服务
│   └── rag_tools_mcp.py   # RAG工具MCP暴露
├── tests/                 # 测试
└── memory_store/          # 持久化记忆存储
```

---

## 2. 配置层：config.py

### 2.1 模型配置

```python
LLM_MODEL = "qwen2.5:3b"      # 主生成模型（本地Ollama部署）
EMBED_MODEL = "bge-m3"         # 嵌入模型（中文检索优化）
LLM_TEMP = 0.1                 # LLM温度（低温度保证事实准确性）
```

### 2.2 图像生成配置（火山引擎即梦AI）

```python
JIMENG_IMAGE_ENABLED = False   # 是否启用图像生成
JIMENG_MODEL = "jimeng_t2i_v31"  # 文生图模型
JIMENG_IMAGE_OUTPUT_DIR = "./literature_output/face_images"
```

### 2.3 检索配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| CHUNK_SIZE | 120 | 文档分块大小 |
| CHUNK_OVERLAP | 15 | 分块重叠量 |
| RETRIEVE_TOP_K | 3 | 检索返回条数 |
| BM25_TOP_K | 7 | BM25召回条数 |
| VECTOR_TOP_K | 7 | 向量召回条数 |
| HYBRID_FINAL_K | 10 | 混合后最终返回条数 |
| VECTOR_WEIGHT | 0.6 | 向量分数权重 |
| BM25_WEIGHT | 0.4 | BM25分数权重 |
| ENABLE_HYBRID_SEARCH | True | 混合检索开关 |

### 2.4 缓存与日志

- **Redis缓存**：检索结果缓存 TTL=100s，问答缓存 TTL=100s
- **日志**：支持 DEBUG/INFO/WARNING/ERROR 四级，可写本地文件，最大10MB/文件滚动

---

## 3. RAG 检索管线：rag/ 模块

### 3.1 混合检索（rag/vectorstore/hybrid_search.py）

核心入口函数 `hybrid_retrieve(query, skip_cache)`，执行流程：

```
1. 读取缓存（Redis）
   ├── 命中 → 直接返回
   └── 未命中 ↓
2. 双路召回
   ├── 向量检索 (ChromaDB MMR) → vec_docs
   └── BM25关键词检索 → bm25_pairs
3. 分数融合
   ├── 向量分数归一化（位置加权）
   ├── BM25分数归一化（最小-最大归一化）
   └── 加权融合：score = vec_norm * 0.6 + bm25_norm * 0.4
4. LLM精排（Reranker）
   ├── 候选集 = 融合结果 TOP (HYBRID_FINAL_K * 2)
   └── 精排后取 TOP HYBRID_FINAL_K
5. 写缓存 → 返回
```

**关键设计**：
- **降级策略**：BM25异常时自动降级为纯向量检索，精排失败时降级为融合排序
- **缓存跳过**：反思重试时 `skip_cache=True`，强制真实检索避免脏缓存
- **异常体系**：`VectorStoreException`、`BM25IndexException` 分类捕获

### 3.2 LLM 精排器（rag/reranker/llm_reranker.py）

`LLMReranker` 类实现两阶段精排：

1. **关键词预排**（轻量 guard）：query 与 doc 的重叠 token 数降序排列
   - 多粒度分词：英文整体 + 中文单字 + 中文二元组
2. **LLM 精排**：逐条文档用 LLM 打分（0-10），失败时关键词分数兜底

```python
class LLMReranker:
    def rerank(query, docs, top_k) -> List[Document]:
        keyword_ranked = self._keyword_rerank(query, docs)  # 预排
        scored = self._llm_score(query, keyword_ranked)      # LLM打分
        scored.sort(key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in scored[:top_k]]
```

### 3.3 Agentic RAG（rag/agentic/）

**AutoRetriever**：自动检索重试管理器，组合 QueryRewriter + hybrid_retrieve + DocumentGrader。

```python
class AutoRetriever:
    def retrieve(query, history) -> Dict:
        # 1. 查询改写（基于对话历史）
        # 2. 检索 + 文档评分过滤
        # 3. 若相关文档不足 → 改写查询重试（最多 max_retries 次）
        # 4. 去重返回
```

**QueryRewriter**：基于对话历史改写用户查询，提升检索召回率。

**DocumentGrader**：对检索文档进行相关性二分类（relevant/irrelevant），过滤无关文档。

---

## 4. Agent 智能体核心：agent/ 模块

### 4.1 单 Agent（agent/graph_base.py）

基于 LangGraph 的 **Plan-Execute-Reflect-Generate** 四节点流程图：

```
┌──────────────┐     tool=none    ┌──────────────────┐
│ plan_tool_call│ ───────────────→ │ generate_final_ans│
│   (工具规划)  │                  │   (生成最终回答)    │
└──────┬───────┘                  └──────────────────┘
       │ tool=search_knowledge_base          ↑
       ↓                          sufficient │
┌──────────┐    ┌──────────────┐            │
│ run_tool  │───→│ reflect_check │───────────┘
│ (执行工具) │    │  (反思校验)    │
└──────────┘    └──────┬───────┘
                       │ insufficient → plan_tool_call (重试)
```

**各节点详解**：

1. **plan_tool_call**（工具规划）：
   - 分析用户问题，决定是否需要调用知识库检索
   - 接入意图解析结果（core_task 最高优先级）
   - 注入上次反思诊断报告（差异化重规划）
   - 日常闲聊关键词（你好/谢谢/再见等）直接跳过工具调用

2. **run_tool**（执行工具）：
   - 调用 `knowledge_tool.invoke(params)` → 触发混合检索
   - 重试时**累积检索结果**（而非覆盖），信息池不断增长
   - 记录 `last_query` 供下一轮反思禁止重复

3. **reflect_check**（反思校验）：
   - 输出**结构化诊断报告**（JSON）：
     ```json
     {
       "sufficient": false,
       "missing_aspects": ["缺少行当分类细节"],
       "suggested_queries": ["京剧 生旦净丑 分类"],
       "reason": "资料仅覆盖剧目名称，未涉及行当分类"
     }
     ```
   - 最多重试 `MAX_REFLECT_TIMES=2` 次

4. **generate_final_ans**（生成最终回答）：
   - 整合工具检索结果 + 记忆上下文 + 意图约束
   - 历史对话只保留最近2条用户消息摘要（旧 AI 输出不带入）

**Trace 追踪**：`agent/trace.py` 提供全链路执行追踪，记录每一步的耗时和状态。

### 4.2 多 Agent（agent/multi_agent.py）

基于 LangGraph 的 **Supervisor-Worker** 主管-工人模式：

```
                        ┌─────────────┐
                        │ supervisor   │ 意图识别与任务分派
                        │   _node      │
                        └──────┬──────┘
                               │ worker_route()
              ┌────────────────┼────────────────┐
              ↓                ↓                 ↓
    ┌────────────┐   ┌────────────┐    ┌────────────┐
    │search_worker│   │lyrics_worker│   │face_worker │ ... (6 workers)
    └─────┬──────┘   └─────┬──────┘    └─────┬──────┘
          │                │                  │
          └────────────────┼──────────────────┘
                           │ compound_worker_finish
                           ↓
                    ┌──────────────┐
                    │summary_agent │ 汇总生成回复
                    └──────────────┘
```

**Worker 清单**：

| Worker | 功能 | 对应业务模块 |
|--------|------|-------------|
| search_worker | 知识库检索 | rag.vectorstore.hybrid_retrieve |
| lyrics_worker | 戏词解剖 | opera.lyrics.annotate_lyrics |
| character_worker | 戏中人对谈 | opera.character.character_chat |
| quiz_worker | 知识闯关出题 | opera.quiz.generate_quiz / generate_quiz_batch |
| guide_worker | 学戏路线 | opera.guide.generate_course |
| face_worker | 脸谱画像 | opera.face.generate_face_profile |

**复合需求支持**：当用户请求"介绍一下曹操并生成脸谱"时，supervisor 可同时分派 search_worker + face_worker，两个 worker 串行执行后汇总。

**反思重试**：search_worker 结果不足时自动触发 supervisor 重新规划（最多 2 次），每次使用不同的检索策略。

### 4.3 意图解析层（agent/intent.py）

`parse_intent(query)` 函数负责解析用户请求，输出结构化意图：

```python
{
    "core_task": "知识问答",           # 核心任务
    "must_require": ["基于知识库"],    # 必须满足的要求
    "forbid_list": ["禁止编造"],       # 禁止行为
    "reference_material_required": True # 是否强制基于文档
}
```

### 4.4 意图校验守卫（agent/intent_guard.py）

纯函数校验层，轻量可单测：

- `filter_tasks_by_intent`：根据意图过滤不允许的 worker
- `is_knowledge_query`：判断是否为知识问答
- `validate_worker_against_query`：校验 worker 与用户请求的匹配度
- `filter_history_recent`：历史对话过滤，只取最近用户消息
- `normalize_face_image_result`：规范化脸谱图片路径
- `is_image_intent`：检测用户是否有图像诉求

### 4.5 输出校验层（agent/output_validator.py）

在生成结果返回给用户前进行最终校验：

1. **意图对齐检查**：生成内容是否满足 core_task / must_require / forbid_list
2. **图像诉求检查**：用户要求图片时是否真正生成了图片
3. **文档约束检查**：是否严格基于参考文档生成
4. **失败自动重试**：最多 `MAX_VALIDATE_RETRY` 次

### 4.6 任务上下文隔离（agent/task_context.py）

每次新提问创建独立 task_id，隔离 Episodic 记忆层，杜绝跨任务污染。

### 4.7 LLM 调用工具（agent/llm_utils.py）

`invoke_with_retry(prompts, model, temperature, timeout, task_name)`：
- 统一超时控制（默认 120s）
- 最大重试 2 次
- 全链路日志记录
- 超时或重试耗尽抛出 `LLMModelException`

---

## 5. Opera 戏曲业务：opera/ 模块

### 5.1 戏词解剖室（opera/lyrics.py）

`annotate_lyrics(lyrics)` 对用户输入的戏词进行多维度解读：

```python
{
    "original": "原文",
    "annotation": "逐句白话翻译",
    "allusion": "典故与出处考据",
    "character_mood": "人物心境解读",
    "singing_style": "唱腔段式标注（行当/声腔/曲牌）",
    "appreciation": "60-100字品鉴文案",
    "sources": "知识库参考资料"
}
```

**关键设计**：
- 使用容错 JSON 解析器 `robust_json_loads`（修复 LLM 输出中的全角标点/缺少逗号等问题）
- JSON 硬性规范提示词：要求使用英文双引号、英文半角逗号、转义换行符

### 5.2 戏中人对谈（opera/character.py）

`character_chat(character_id, user_message, session_id, history)` 实现与戏曲人物的角色化对话：

**内置人物档案库**（5个角色）：

| ID | 角色 | 行当 | 性格关键词 |
|----|------|------|-----------|
| muguiying | 穆桂英 | 刀马旦 | 豪爽果敢、智勇双全 |
| baisuzhen | 白素贞 | 青衣 | 温婉深情、外柔内刚 |
| caocao | 曹操 | 花脸(净) | 雄才大略、多疑善忌 |
| dushiniang | 杜十娘 | 青衣 | 刚烈决绝、敢爱敢恨 |
| baozheng | 包拯 | 花脸(净·黑头) | 铁面无私、刚正不阿 |

**对话流程**：
1. 检索知识库 + 用户记忆上下文
2. 组装历史对话（限制最近 8 条）
3. 构建角色人格提示词（性格/口吻/经典台词/背景知识）
4. LLM 生成角色口吻回复
5. 记录用户画像到记忆系统

### 5.3 知识闯关（opera/quiz.py）

**单题生成** `generate_quiz(topic, difficulty, session_id)`：
- 4 个难度级别：小白 / 入门 / 票友 / 老戏骨
- 5 个主题：行当 / 经典剧目 / 戏曲流派 / 戏曲历史 / 戏曲术语
- 按主题检索知识库作为出题素材
- 解析失败自动降级为模板题库

**动态批量出题** `generate_quiz_batch(topic, difficulty, session_id, count, reference_text, reference_required)`：
- 支持用户指定任意题目数量
- 支持"严格基于文档"模式：出题后自动校验知识点来源，不匹配自动重生成（最多 3 次）
- 支持参考文档约束（`reference_required=True`）

**判题** `check_answer(question_id, user_answer, correct_answer, explanation, session_id)`：
- 大小写不敏感匹配
- 返回对/错判定 + 知识点讲解 + 鼓励文案

### 5.4 学戏路线（opera/guide.py）

`generate_course(topic, days, session_id, level)` 生成阶梯式学戏课程：

- 最大 14 天，默认 7 天
- 每天包含：标题 + 学习内容（80-150字）+ 互动问题
- 课程循序渐进：Day1 基础 → 中间深入 → 最后综合回顾
- 记录到任务级记忆（Episodic）

`get_course_progress(session_id)` 从会话时序记忆中读取学习进度。

### 5.5 脸谱画像（opera/face.py）

核心功能：根据用户喜好生成专属"戏曲人格脸谱"解读卡片 + 即梦AI图像。

**文本版**：LLM 生成脸谱设计描述

```python
{
    "face_name": "脸谱名号（2-4字）",
    "color": "主色（红/黑/白/蓝/绿/黄/紫/金/银）",
    "color_meaning": "颜色含义",
    "pattern": "图案设计描述",
    "matching_character": "对应戏曲人物",
    "personality_text": "40-80字人格解读",
    "share_card": "60-100字分享卡片文案"
}
```

**图像版**：基于火山引擎即梦AI（视觉智能CV服务）

- 双 SDK 兼容：新版 `volcenginesdkcv20240606` + 旧版 `volcengine.visual`
- 流程：构造 Prompt → Text2ImgXLSft 文生图 → 解析图片URL → 下载到本地
- 多形态响应解析：兼容对象属性访问 / to_dict() / dict 三种形态
- 任何异常都不影响文本版交付，自动降级

**颜色含义内置知识**：红→忠勇 / 黑→刚正 / 白→奸诈 / 蓝→刚强 / 绿→勇猛 / 黄→暴烈 / 紫→稳重 / 金→神佛 / 银→妖精

---

## 6. 文献生成：literature/ 模块

### 6.1 LiteratureGenerator 类（literature/generator.py）

**核心方法**：

| 方法 | 功能 |
|------|------|
| `generate_content(genre, theme, length)` | LLM 生成戏曲文献文本 |
| `save_as_txt(content, file_path)` | 输出纯文本 |
| `save_as_md(content, file_path, genre, theme)` | 输出带标题结构的 Markdown |
| `save_as_pdf(content, file_path, genre, theme)` | 输出中文 PDF（reportlab） |
| `generate_literature(genre, theme, length, formats)` | 统一入口，一次生成多种格式 |
| `list_literature_files()` | 列出所有已生成文献 |
| `read_literature_content(relative_path)` | 读取文献预览 |

**PDF 生成亮点**：
- 自动检测系统中文字体（Windows/macOS/Linux）
- 多层降级：微软雅黑→宋体→黑体→PingFang→WQY→Helvetica
- XML 特殊字符转义防止 ReportLab 解析报错

**安全设计**：
- `read_literature_content` 进行路径越权检查（`os.path.normpath` + 前缀验证）
- 超时保护：180 秒超时返回明确错误

---

## 7. 评估体系：evaluation/ 模块

### 7.1 独立评估 Agent（evaluation/eval_agent.py）

使用独立 LLM（默认 `deepseek-r1:7b`）对生成 Agent 的输出进行深度评估，打破"自己评自己"的盲区。

**5 步评估流程**：

```
Step 1: 事实声明提取  →  从生成文本中提取所有事实性声明
Step 2: 事实核查       →  逐条检索知识库验证（verified/partial/unverified/contradicted）
Step 3: 意图对齐检查   →  检查是否完全对齐用户 core_task/must_require/forbid_list
Step 4: 质量评分       →  7维度综合评分（1-5分制）
Step 5: 决策路由       →  pass / retry / fail
```

**7 维度评分体系**：

| 维度 | 说明 |
|------|------|
| factual_accuracy | 事实准确性 |
| intent_alignment | 意图对齐度 |
| completeness | 完整性 |
| hallucination_free | 无幻觉率 |
| reference_accuracy | 引用准确性 |
| task_completion | 任务完成度 |
| user_satisfaction | 用户满意度预估 |

**决策矩阵**：

| 条件 | 动作 |
|------|------|
| 事实准确率 ≥ 0.8 且综合 ≥ 4.0 | pass |
| 有 contradicted | retry |
| 意图对齐失败 | retry |
| 图像缺失 | retry |
| 事实准确率 < 0.6 | fail |

**Critic-Refine 循环**（核心创新）：

```
1. 快速预筛选 → 2. Span级幻觉检测 → 3. 修正指令生成 → 4. 精确重生成 → 5. 最多EVAL_MAX_RETRY轮
```

- **Span 级幻觉检测**：一次 LLM 调用输出所有幻觉的精确位置和修正内容
- **批量事实核查**：一次 LLM 调用核查所有声明，效率优于逐条核查
- **修正指令生成**：将评估诊断转化为可执行的修正操作列表

### 7.2 检索指标评测（evaluation/metrics_evaluator.py）

动态生成测评问题，计算标准检索指标：

| 指标 | 说明 |
|------|------|
| Recall@K | 前K个结果中相关文档的召回率 |
| HitRate@K | 前K个结果中至少命中一个相关文档的比例 |
| MRR@K | 平均倒数排名 |
| NDCG@K | 归一化折损累计增益 |

### 7.3 快速预筛选（evaluation/pre_filter.py）

`quick_check(query, text, image_resources)` 轻量规则检查：
- 强制图片需求：关键词包含"生成/画/绘制/图片/图像/脸谱"
- 空文本检测
- 避免不必要的深度评估开销

---

## 8. API 服务层：api/ 模块

### 8.1 应用入口（api/main.py）

**生命周期管理**：
1. **启动阶段**：初始化 Redis → 重建 BM25 索引 → 构建单/多 Agent 实例 → 确认文献目录 → 初始化分层记忆
2. **关闭阶段**：关闭 Redis 连接

**路由注册**：

| 路由前缀 | 功能 |
|----------|------|
| /api/chat/* | 对话接口（普通/多Agent/流式） |
| /api/kb/* | 知识库管理（入库/检索/统计/清空/上传） |
| /api/literature/* | 文献生成与管理 |
| /api/evaluation/* | 检索指标评测 |
| /api/opera/* | 戏曲科普（戏词/人物/闯关/路线/脸谱） |
| /api/health | 健康检查 |
| /api/cache/* | 缓存管理 |
| /api/log/* | 日志查询 |

**全局异常处理**：
- `BaseRAGException` → 业务异常（code + msg + detail）
- `Exception` → 兜底未知异常（code=9999）

**静态文件服务**：`/static` 挂载 `literature_output` 目录，前端可直接访问生成的脸谱图片。

### 8.2 对话路由（api/routes/chat_routes.py）

**普通对话** `POST /api/chat/normal`：
1. 检查问答缓存
2. 解析用户意图
3. 创建任务隔离（task_id）
4. 调用单 Agent 实例
5. 输出校验
6. 保存会话记忆 + 写缓存

**多智能体对话** `POST /api/chat/multi_agent`：
1. 检查缓存
2. 解析意图 + 任务隔离
3. 调用多 Agent 实例
4. **输出校验 + 图像缺失重试**：最多重试 `MAX_VALIDATE_RETRY` 次
5. 保存记忆 + 写缓存
6. 合并图片资源到返回报文

**流式对话** `POST /api/chat/stream`：
- SSE 协议推送 token
- 实时流式响应

### 8.3 速率限制（api/limiter.py）

使用 slowapi 实现基于 IP 的速率限制：
- 普通对话：10次/分钟
- 多智能体：8次/分钟
- 流式对话：15次/分钟

---

## 9. 前端展示层：frontend/ 模块

### 9.1 API 客户端（frontend/api_client.py）

`ApiClient` 类封装所有后端接口的 HTTP 调用，统一错误处理（连接失败/超时/HTTP错误）。

支持的接口方法：
- 对话：`chat_normal`、`chat_multi_agent`
- 知识库：`kb_ingest`、`kb_retrieve`、`kb_stats`、`kb_upload`
- 文献：`literature_generate`、`literature_list`、`literature_read`
- 评测：`evaluation_run`、`evaluation_report`
- 戏曲科普：`lyrics_annotate`、`character_chat`、`quiz_generate`、`course_generate`、`face_generate`
- 系统：`health_check`、`cache_status`、`cache_clear`

### 9.2 Streamlit 页面

| 页面文件 | Emoji | 功能 |
|----------|-------|------|
| 1_💬_智能对话.py | 💬 | 普通问答 + 多Agent协作 |
| 5_📜_戏词解剖室.py | 📜 | 戏词多维度解读 |
| 6_🎭_戏中人对谈.py | 🎭 | 与戏曲人物角色对话 |
| 7_🎮_知识闯关.py | 🎮 | 戏曲知识答题 |
| 8_🧭_个性化学戏路线.py | 🧭 | 学戏课程规划 |
| 9_🎨_脸谱画像.py | 🎨 | 脸谱人格解读 + AI图像 |

---

## 10. 工具与基础设施：tools/、utils/ 模块

### 10.1 工具注册（tools/custom_tools.py）

注册单 Agent 可用的 `StructuredTool`：

```python
knowledge_tool = StructuredTool.from_function(search_knowledge_base)
```

工具内部调用 `hybrid_retrieve` 混合检索，异常直接向上抛出（由 Agent 层的异常处理捕获）。

### 10.2 异常体系（utils/rag_exceptions.py）

统一的业务异常类，继承 `BaseRAGException`：

| 异常类 | code | 说明 |
|--------|------|------|
| VectorStoreException | 1001 | 向量库异常 |
| BM25IndexException | 1002 | BM25索引异常 |
| LLMModelException | 1003 | LLM调用异常 |
| AgentFlowException | 1004 | Agent流程异常 |
| DocProcessException | 1005 | 文档处理异常 |
| CacheSerializeException | 1006 | 缓存序列化异常 |

### 10.3 日志系统（utils/logger.py）

- 四级日志：DEBUG / INFO / WARNING / ERROR
- `log_info(module, message)` 标准化日志格式
- 支持文件输出 + 滚动分割（最大 10MB，保留 7 个备份）

### 10.4 缓存工具（utils/cache_utils.py）

基于 Redis 的两层缓存：
- **检索文档缓存**：`rag:ret:{query_hash}`，TTL=100s
- **问答缓存**：`rag:chat:{session_id}:{query_hash}`，TTL=100s

支持文档序列化/反序列化、脏数据自动清理。

### 10.5 JSON 修复器（utils/json_repair.py）

`robust_json_loads(content)` 容错解析 LLM 输出的 JSON：
- 修复全角标点（，、：→ , :）
- 修复缺失逗号
- 修复值内换行符
- 修复多余引号
- 提取 JSON 对象/数组片段

### 10.6 全局异常处理（utils/exception_handler.py）

`@global_exception_handler` 装饰器，捕获节点函数中的：
- `httpx.ConnectError` → 抛出 `LLMModelException`
- 其他异常 → 抛出 `AgentFlowException`

---

## 11. MCP 服务：mcp_server/ 模块

### 11.1 RAG 工具 MCP 服务（mcp_server/rag_tools_mcp.py）

将知识库检索能力暴露为 MCP（Model Context Protocol）服务，供外部 AI 客户端调用。

---

## 12. 记忆系统：agent/memory/ 模块

### 12.1 三层记忆架构

```
┌─────────────────────────────────┐
│       Permanent Memory          │  永久记忆（人物档案、系统知识）
│       permanent_memory.py       │  JSON 文件持久化，不会过期
└───────────────┬─────────────────┘
                │
┌───────────────┴─────────────────┐
│       Episodic Memory           │  情节记忆（任务级，按 task_id 隔离）
│       episodic_memory.py        │  Redis 存储，每次新提问创建新 task
└───────────────┬─────────────────┘
                │
┌───────────────┴─────────────────┐
│       Session Memory            │  会话时序记忆（会话级，按 session_id 隔离）
│       session_memory.py         │  Redis 存储，自动清理过期会话
└─────────────────────────────────┘
```

**记忆管理器**（`agent/memory/__init__.py`）统一对外接口：
- `record_dialogue(session_id, role, content)`：记录对话
- `record_thinking(session_id, content)`：记录思考过程
- `record_tool_call(session_id, tool_name, params)`：记录工具调用
- `record_tool_result(session_id, tool_name, result)`：记录工具结果
- `format_memory_context(query, session_id)`：格式化记忆上下文供 prompt 使用
- `task.create_task(task_id, title, description, objectives)`：创建任务

### 12.2 任务隔离策略

每次新提问 = 全新 task_id，Episodic 记忆严格按 task_id 隔离，杜绝跨任务污染旧输出。

---

## 13. 知识库构建：generate_opera_kb.py

知识库构建脚本，将戏曲知识文档分块嵌入 ChromaDB：

1. 加载文档目录（`knowledge_base/opera/`）
2. 文档分块（chunk_size=120, overlap=15）
3. 生成嵌入向量（bge-m3）
4. 存入 ChromaDB（持久化路径 `./chroma_db`）
5. 构建 BM25 关键词索引

---

## 附录：关键技术决策

| 决策 | 原因 |
|------|------|
| 本地 Ollama 部署 | 零成本、数据隐私、离线可用 |
| bge-m3 嵌入模型 | 中文检索效果优于通用模型 |
| BM25+向量混合检索 | 兼顾关键词匹配与语义理解 |
| LLM 精排（Reranker） | 二次精排提升 Top-K 相关性 |
| 纠错式反思 | 结构化诊断报告驱动差异化重规划 |
| 主管-工人多 Agent | 复杂任务拆分，专业分工 |
| 任务隔离记忆 | 杜绝跨任务污染旧输出 |
| 独立评估 Agent | 打破"自己评自己"盲区 |
| 容错 JSON 解析 | 处理 LLM 输出不稳定问题 |
| Redis 双层缓存 | 减少重复检索和 LLM 调用 |