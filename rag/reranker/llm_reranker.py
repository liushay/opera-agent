# rag/reranker/llm_reranker.py LLM 精排器
# 功能：
#   1. 对 BM25+向量融合后的候选文档进行二次精排
#   2. 使用 ChatOllama 对每个候选文档与查询进行相关性评分
#   3. 输出重排序后的文档列表（保持原 Document 结构）
# 设计：
#   - 提供 LLM 评分（cross-encoder 风格）与关键词降级两种策略
#   - 精排失败自动降级为原始顺序，不阻断主流程
from typing import List, Optional
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import json
import re

import config
from utils.logger import log_info, log_warn, log_error


class LLMReranker:
    """基于 LLM 的文档精排器"""

    def __init__(
        self,
        model: str = None,
        temperature: float = 0.0,
        top_k: int = None,
        enable: bool = True,
    ):
        """
        Args:
            model: Ollama 模型名，默认 config.LLM_MODEL
            temperature: 评分模型温度（精排需要低温度）
            top_k: 精排后返回条数，默认 None 表示全部
            enable: 是否启用精排（False 则直接返回输入）
        """
        self.model = model or config.LLM_MODEL
        self.temperature = temperature
        self.top_k = top_k
        self.enable = enable
        self._llm = None

    def _get_llm(self) -> ChatOllama:
        """懒加载 LLM，避免模块导入时连接失败"""
        if self._llm is None:
            self._llm = ChatOllama(model=self.model, temperature=self.temperature)
        return self._llm

    # ===================== 公开入口 =====================

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        对候选文档进行精排
        Args:
            query: 用户查询
            docs: 候选文档列表（BM25+向量融合后）
            top_k: 返回条数（覆盖实例默认值）
        Returns:
            精排后的文档列表
        """
        if not self.enable:
            return docs[: top_k or self.top_k] if (top_k or self.top_k) else docs

        if not docs:
            return docs

        # 1. 先使用轻量关键词打分作为 guard，避免 LLM 全量调用耗时
        keyword_ranked = self._keyword_rerank(query, docs)
        log_info(
            "LLM精排",
            f"关键词预排完成，候选{len(docs)}条，开始LLM精排",
        )

        # 2. 使用 LLM 对每条文档与查询的相关性评分
        scored = self._llm_score(query, keyword_ranked)

        # 3. 按分数降序排列
        scored.sort(key=lambda x: x["score"], reverse=True)

        final_k = top_k or self.top_k or len(scored)
        result = [item["doc"] for item in scored[:final_k]]

        log_info(
            "LLM精排",
            f"精排完成，返回{len(result)}条（原始{len(docs)}条）",
        )
        return result

    # ===================== 关键词降级精排 =====================

    def _keyword_rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """
        关键词重叠度精排（轻量，作为 LLM 精排的 guard/降级方案）
        按 query 与 doc 的重叠 token 数降序排列
        """
        query_terms = set(self._tokenize(query))
        if not query_terms:
            return docs

        scored = []
        for doc in docs:
            doc_terms = set(self._tokenize(doc.page_content))
            hits = query_terms & doc_terms
            score = len(hits) / len(query_terms)
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored]

    def _tokenize(self, text: str) -> List[str]:
        """多粒度分词：英文整体 + 中文单字和二元组"""
        tokens = set()
        for m in re.findall(r"[a-zA-Z0-9]+", text.lower()):
            tokens.add(m.lower())
        cn_chars = re.findall(r"[\u4e00-\u9fff]", text)
        if cn_chars:
            tokens.update(cn_chars)
            for i in range(len(cn_chars) - 1):
                tokens.add(cn_chars[i] + cn_chars[i + 1])
        return list(tokens)

    # ===================== LLM 评分 =====================

    def _llm_score(self, query: str, docs: List[Document]) -> List[dict]:
        """
        使用 LLM 为每条文档打分（0-10整数），若 LLM 不可用则使用关键词分数兜底
        """
        try:
            llm = self._get_llm()
            scored_items = []
            for doc in docs:
                text = doc.page_content[:500]  # 截断，控制 token
                prompt = f"""你是文档相关性评分器。请判断以下文档与用户查询的相关程度。
用户查询：{query}

文档内容：
{text}

请仅输出一个 0-10 的整数，表示文档与查询的相关性（10=完全相关，0=完全无关）。
输出格式：仅整数
"""
                try:
                    resp = llm.invoke([HumanMessage(content=prompt)])
                    score = self._parse_score(resp.content)
                except Exception as e:
                    log_warn("LLM精排", f"单条文档LLM评分失败，使用关键词分数：{e}")
                    score = self._keyword_score(query, doc)
                scored_items.append({"doc": doc, "score": score})
            return scored_items
        except Exception as e:
            log_warn("LLM精排", f"LLM精排整体失败，使用关键词分数兜底：{e}")
            return [
                {"doc": doc, "score": self._keyword_score(query, doc)}
                for doc in docs
            ]

    def _parse_score(self, content: str) -> float:
        """解析 LLM 输出为 0-10 分数，容错处理"""
        content = content.strip()
        # 提取第一个数字
        match = re.search(r"(\d+(?:\.\d+)?)", content)
        if match:
            score = float(match.group(1))
            return max(0.0, min(10.0, score))
        return 0.0

    def _keyword_score(self, query: str, doc: Document) -> float:
        """关键词重叠度分数（0-1 映射到 0-10）"""
        query_terms = set(self._tokenize(query))
        doc_terms = set(self._tokenize(doc.page_content))
        if not query_terms:
            return 0.0
        hits = query_terms & doc_terms
        return (len(hits) / len(query_terms)) * 10.0


# 全局单例（默认开启，使用配置中的 LLM 模型）
reranker = LLMReranker(
    model=config.LLM_MODEL,
    temperature=0.0,
    top_k=config.HYBRID_FINAL_K,
    enable=True,
)