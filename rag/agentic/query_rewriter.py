# rag/agentic/query_rewriter.py 查询改写器
# 功能：将多轮对话中的模糊/指代式问题改写为独立可检索的查询
# 例："它的原理是什么？" → 结合历史改写为 "BM25关键词检索的原理是什么？"
from typing import Optional, List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

import config
from utils.logger import log_info, log_warn


class QueryRewriter:
    """基于 LLM 的查询改写器"""

    def __init__(self, model: str = None, temperature: float = 0.1):
        self.model = model or config.LLM_MODEL
        self.temperature = temperature
        self._llm = None

    def _get_llm(self) -> ChatOllama:
        if self._llm is None:
            self._llm = ChatOllama(model=self.model, temperature=self.temperature)
        return self._llm

    def rewrite(
        self,
        query: str,
        history: Optional[List[dict]] = None,
        max_retry: int = 1,
    ) -> str:
        """
        改写查询
        Args:
            query: 用户当前问题
            history: 历史对话 [{role: "user"/"ai", content: "..."}]
            max_retry: LLM 输出异常时重试次数
        Returns:
            改写后的独立查询
        """
        # 无历史时直接返回原查询（多轮对话场景才需要改写）
        if not history:
            return query

        # 构造历史文本
        history_text = "\n".join(
            f"{'用户' if h.get('role') == 'user' else 'AI'}: {h.get('content', '')}"
            for h in history[-6:]  # 最多取最近 6 轮
        )

        prompt = f"""你是查询改写助手。请根据历史对话上下文，将用户最新问题改写为一条**独立的、可检索知识库的完整查询**。
要求：
1. 将指代词（它、这个、那个、那等）扩展为完整概念
2. 保留原始查询的核心意图
3. 仅输出改写后的查询，不要输出解释

历史对话：
{history_text}

用户最新问题：{query}

改写后的查询：
"""
        for attempt in range(max_retry + 1):
            try:
                llm = self._get_llm()
                resp = llm.invoke([HumanMessage(content=prompt)])
                rewritten = resp.content.strip()
                if rewritten:
                    log_info("查询改写", f"改写成功：'{query}' → '{rewritten}'")
                    return rewritten
            except Exception as e:
                log_warn("查询改写", f"第{attempt + 1}次改写失败：{e}")
        # 全部失败时原样返回
        log_warn("查询改写", f"改写失败，返回原查询：'{query}'")
        return query


# 全局单例
query_rewriter = QueryRewriter()