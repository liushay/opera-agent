# rag/agentic/auto_retriever.py 自动检索重试管理器（Agentic RAG 核心）
# 功能：
#   1. 组合 QueryRewriter + hybrid_retrieve + DocumentGrader
#   2. 检索文档不足时自动改写查询重试（最多 N 次）
#   3. 返回经文档相关性过滤后的最终文档
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

import config
from rag.vectorstore.hybrid_search import hybrid_retrieve
from rag.agentic.query_rewriter import query_rewriter
from rag.agentic.document_grader import document_grader
from utils.logger import log_info, log_warn


class AutoRetriever:
    """自动检索重试管理器"""

    def __init__(
        self,
        max_retries: int = 2,
        min_relevant_docs: int = 1,
        grade_mode: str = "binary",
    ):
        self.max_retries = max_retries
        self.min_relevant_docs = min_relevant_docs
        self.grade_mode = grade_mode

    def retrieve(
        self,
        query: str,
        history: Optional[List[dict]] = None,
        use_grader: bool = True,
        use_rewriter: bool = True,
    ) -> Dict[str, Any]:
        """
        执行 Agentic 检索（自动改写+检索+文档评分+重试）
        Args:
            query: 用户原始查询
            history: 历史对话（用于查询改写）
            use_grader: 是否使用文档相关性过滤
            use_rewriter: 是否使用查询改写
        Returns:
            {
              "query": 最终使用的查询,
              "docs": 最终过滤后的文档列表,
              "all_docs": 重试过程中所有检索到的文档,
              "retry_count": 重试次数,
              "rewritten": 是否发生改写,
              "trace": [{round, query, doc_count, relevant_count}]
            }
        """
        trace = []
        all_docs = []
        final_query = query
        rewritten = False

        # 1. 查询改写（首轮）
        if use_rewriter and history:
            rewritten_query = query_rewriter.rewrite(query, history)
            if rewritten_query and rewritten_query != query:
                final_query = rewritten_query
                rewritten = True
                log_info("自动检索", f"查询改写：'{query}' → '{final_query}'")

        # 2. 检索 + 评分 + 重试循环
        current_query = final_query
        relevant_docs = []

        for attempt in range(self.max_retries + 1):
            # 检索
            docs = hybrid_retrieve(current_query)
            all_docs.extend(docs)

            # 文档评分过滤
            if use_grader and docs:
                graded = document_grader.grade(
                    current_query, docs, mode=self.grade_mode
                )
                relevant_docs = [g["doc"] for g in graded if g["relevant"]]
                relevant_count = len(relevant_docs)
            else:
                relevant_docs = docs
                relevant_count = len(docs)

            trace.append({
                "round": attempt + 1,
                "query": current_query,
                "doc_count": len(docs),
                "relevant_count": relevant_count,
            })
            log_info(
                "自动检索",
                f"第{attempt + 1}轮：查询'{current_query}'，召回{len(docs)}条，相关{relevant_count}条",
            )

            # 判断是否足够
            if relevant_count >= self.min_relevant_docs or attempt >= self.max_retries:
                break

            # 重试：改写查询（无历史时用"更聚焦"的重写）
            if attempt < self.max_retries:
                new_query = query_rewriter.rewrite(
                    current_query,
                    history=[{"role": "user", "content": query}],
                )
                if new_query and new_query != current_query:
                    current_query = new_query
                else:
                    # 改写失败时增加关键词扩充提示
                    current_query = f"{query} 特点 原理 是什么"
                log_info("自动检索", f"重试改写：'{current_query}'")

        # 去重（按内容）
        seen = set()
        dedup_docs = []
        for doc in (relevant_docs or docs):
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                dedup_docs.append(doc)

        result = {
            "query": current_query,
            "docs": dedup_docs,
            "all_docs": all_docs,
            "retry_count": len(trace) - 1,
            "rewritten": rewritten,
            "trace": trace,
        }
        log_info("自动检索", f"最终返回{len(dedup_docs)}条文档，重试{result['retry_count']}次")
        return result


# 全局单例
auto_retriever = AutoRetriever(
    max_retries=config.MAX_REFLECT_TIMES,
    min_relevant_docs=1,
    grade_mode="binary",
)