# rag/agentic/document_grader.py 文档相关性评分器
# 功能：判断检索到的文档是否与用户问题相关、是否足够回答
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

import config
from utils.logger import log_info, log_warn


class DocumentGrader:
    """基于 LLM 的文档相关性评分器"""

    BINARY_PROMPT = """你是文档相关性评估专家。请判断以下文档是否与用户问题**相关**。

用户问题：{query}

文档内容：
{document}

请仅输出一个单词：yes 或 no
- yes = 文档内容与问题相关，包含回答问题所需的关键信息
- no = 文档内容与问题无关或相关性极低

输出：
"""

    RATING_PROMPT = """你是文档相关性评分专家。请对以下文档与用户问题的相关性进行 0-10 整数评分。

用户问题：{query}

文档内容：
{document}

评分标准：
- 10：完全直接相关，包含问题答案的关键信息
- 7-9：高度相关，包含大部分所需信息
- 4-6：部分相关，包含一些相关信息
- 1-3：低相关，仅边缘相关
- 0：完全无关

请仅输出 0-10 的整数：
"""

    def __init__(self, model: str = None, temperature: float = 0.0):
        self.model = model or config.LLM_MODEL
        self.temperature = temperature
        self._llm = None

    def _get_llm(self) -> ChatOllama:
        if self._llm is None:
            self._llm = ChatOllama(model=self.model, temperature=self.temperature)
        return self._llm

    def grade(
        self,
        query: str,
        docs: List[Document],
        mode: str = "binary",
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        对文档进行相关性评分
        Args:
            query: 用户查询
            docs: 候选文档列表
            mode: "binary"（yes/no）或 "rating"（0-10 评分）
            threshold: binary 模式下判定相关的阈值（1=yes）
        Returns:
            [{"doc": Document, "relevant": bool, "score": float}]
        """
        if not docs:
            return []

        results = []
        for doc in docs:
            text = doc.page_content[:600]
            try:
                answer = self._grade_single(query, text, mode)
                if mode == "binary":
                    relevant = answer == "yes"
                    score = 1.0 if relevant else 0.0
                else:
                    score = self._parse_score(answer)
                    relevant = score >= threshold * 10
                results.append({"doc": doc, "relevant": relevant, "score": score})
            except Exception as e:
                log_warn("文档评分", f"单条文档评分失败，默认标记相关：{e}")
                results.append({"doc": doc, "relevant": True, "score": 1.0})

        log_info(
            "文档评分",
            f"评分完成：{sum(1 for r in results if r['relevant'])}/{len(results)}条相关",
        )
        return results

    def _grade_single(self, query: str, document: str, mode: str) -> str:
        """对单条文档执行 LLM 评分"""
        prompt = (
            self.RATING_PROMPT if mode == "rating" else self.BINARY_PROMPT
        ).format(query=query, document=document)
        try:
            llm = self._get_llm()
            resp = llm.invoke([HumanMessage(content=prompt)])
            return resp.content.strip().lower()
        except Exception as e:
            log_warn("文档评分", f"LLM调用失败：{e}")
            raise

    def _parse_score(self, content: str) -> float:
        """解析 LLM 输出的 0-10 分数"""
        import re

        match = re.search(r"(\d+(?:\.\d+)?)", content)
        if match:
            score = float(match.group(1))
            return max(0.0, min(10.0, score))
        return 0.0

    def filter_relevant(
        self,
        query: str,
        docs: List[Document],
        mode: str = "binary",
        threshold: float = 0.5,
    ) -> List[Document]:
        """返回被判定为相关的文档列表（过滤无关文档）"""
        graded = self.grade(query, docs, mode=mode, threshold=threshold)
        return [g["doc"] for g in graded if g["relevant"]]


# 全局单例
document_grader = DocumentGrader()

# 同时提供 alias 名称供 MCP/Agent 使用
DocumentGraderInstance = document_grader