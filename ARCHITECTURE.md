# 🏗️ RAG 知识库智能平台 - 架构设计

## 一、系统总体架构

```mermaid
graph TB
    subgraph 前端层[Streamlit 前端]
        UI1[💬 智能对话]
        UI2[📚 知识库管理]
        UI3[🎭 文献生成]
        UI4[📊 指标评测]
    end

    subgraph API层[FastAPI 后端]
        API1[chat_routes]
        API2[kb_routes]
        API3[literature_routes]
        API4[evaluation_routes]
        API5[system_routes]
    end

    subgraph Agent层[LangGraph Agent]
        SA[单Agent<br/>规划→执行→反思→生成]
        MA[多Agent<br/>主管→工人→汇总]
        MEM[三层记忆<br/>永久/任务/时序]
        TRACE[Agent Trace<br/>可观测性]
    end

    subgraph AgenticRAG层[Agentic RAG]
        QR[QueryRewriter<br/>查询改写]
        DR[DocumentGrader<br/>文档评分]
        AR[AnswerGrader<br/>答案评分]
        AUTO[AutoRetriever<br/>自动重试]
    end

    subgraph 检索层[混合检索 + 精排]
        VEC[Chroma 向量<br/>MMR召回]
        BM25[BM25 关键词<br/>jieba分词]
        FUSE[分数融合<br/>V=0.6 B=0.4]
        RERANK[LLM Reranker<br/>二次精排]
    end

    subgraph 存储层[存储]
        KB[(知识库<br/>戏曲文献)]
        MEMSTORE[(memory_store<br/>JSON分层)]
        REDIS[(Redis<br/>缓存+会话)]
        TRACEDIR[(agent_traces<br/>JSON)]
    end

    UI1 --> API1
    UI2 --> API2
    UI3 --> API3
    UI4 --> API4

    API1 --> SA & MA
    SA --> MEM
    MA --> MEM
    SA --> TRACE
    SA --> AUTO
    AUTO --> QR & DR
    AUTO --> FUSE
    FUSE --> RERANK

    VEC --> FUSE
    BM25 --> FUSE
    VEC --> KB
    BM25 --> KB
    MEM --> MEMSTORE
    TRACE --> TRACEDIR
    API1 --> REDIS
    AR --> API4
```

## 二、分层隔离记忆架构

```mermaid
graph LR
    subgraph 认知抽象[cognitive/统一认知记忆]
        SW[Semantic Memory<br/>语义记忆]
        EP[Episodic Memory<br/>情景记忆]
        WRK[Working Memory<br/>工作记忆]
    end

    subgraph 物理存储[物理存储层 - JSON]
        PM[permanent_memory.json<br/>系统规则/领域规范/用户偏好]
        TM[tasks/{task_id}.json<br/>任务需求/中间产物/约束]
        SM[sessions/{session_id}.json<br/>对话轨迹/工具调用/报错]
    end

    SW --> PM
    EP --> TM
    WRK --> SM

    subgraph 特性[记忆特性]
        F1[全局共享]
        F2[任务隔离/父子嵌套]
        F3[时间戳/自动归档]
    end

    PM --> F1
    TM --> F2
    SM --> F3
```

## 三、Hybrid Search 召回-融合-精排流水线

```mermaid
graph LR
    Q[用户查询] --> V[向量召回 MMR]
    Q --> B[BM25 召回]
    V --> F[分数融合]
    B --> F
    F --> C[候选集 top-K*2]
    C --> R[LLM Reranker 精排]
    R --> OUT[最终结果 top-K]
```

## 四、Agentic RAG 自动检索重试流程

```mermaid
graph TD
    Q[用户查询] --> RW{有历史?}
    RW -- 是 --> QR[QueryRewriter 改写]
    RW -- 否 --> GO[直接检索]
    QR --> GO
    GO --> H[Hybrid Retrieve + Rerank]
    H --> DG{DocumentGrader<br/>文档足够?}
    DG -- 是 --> R[生成回答]
    DG -- 否 --> RC{重试<N?}
    RC -- 是 --> QR
    RC -- 否 --> R
    R --> AG[AnswerGrader<br/>校验忠实度/相关性]
```

## 五、MCP Server 工具

```mermaid
graph TB
    MCP[MCP Client] <-->|JSON-RPC stdio| SRV[MCP Server]
    SRV --> T1[kb_search 知识库检索]
    SRV --> T2[doc_get 文档获取]
    SRV --> T3[lit_generate 文献生成]
    SRV --> T4[eval_rag RAG评测]
```

## 六、代码目录

```
├── agent/
│   ├── graph_base.py          # 单Agent（含Trace集成）
│   ├── multi_agent.py         # 多Agent
│   ├── trace.py               # 🆕 Agent Trace/Observability
│   └── memory/
│       ├── permanent_memory.py    # 永久静态记忆
│       ├── task_memory.py         # 任务级记忆
│       ├── session_memory.py      # 会话时序记忆
│       ├── memory_manager.py      # 内存管理器（旧）
│       └── unified_memory.py      # 🆕 Working/Episodic/Semantic 抽象
├── rag/
│   ├── reranker/              # 🆕 Reranker 精排模块
│   │   ├── llm_reranker.py
│   │   └── __init__.py
│   ├── agentic/               # 🆕 Agentic RAG 组件
│   │   ├── query_rewriter.py  # 查询改写
│   │   ├── document_grader.py # 文档相关性评分
│   │   ├── answer_grader.py   # 答案质量评分
│   │   └── auto_retriever.py  # 自动检索重试
│   └── vectorstore/
│       └── hybrid_search.py   # 已接入 Reranker 精排
├── evaluation/
│   ├── metrics_evaluator.py   # 基础指标（Recall/HitRate/MRR/NDCG）
│   └── advanced_evaluator.py  # 🆕 高级指标（Faithfulness/Relevance/Precision/Recall/TaskSuccess）
├── mcp_server/                # 🆕 MCP Server
│   └── rag_tools_mcp.py
├── tests/                     # 🆕 测试套件
│   ├── test_reranker.py
│   ├── test_agentic.py
│   ├── test_trace.py
│   └── test_mcp.py
└── config.py                  # 新增 Reranker 配置
```

## 七、指标评测体系

| 层级 | 指标 | 说明 |
|------|------|------|
| 基础检索 | Recall@K / HitRate@K / MRR@K / NDCG@K | 检索质量 |
| 高级RAG | Faithfulness | 答案忠实度（无幻觉） |
| 高级RAG | Answer Relevance | 答案是否直接回答 |
| 高级RAG | Context Precision | 相关文档排序靠前程度 |
| 高级RAG | Context Recall | 相关文档是否都被召回 |
| Agent | Agent Task Success | 端到端任务成功率 |