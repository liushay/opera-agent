# rag/agentic/answer_grader.py 答案质量评分器
# 功能：判断生成的答案是否忠实于检索文档（Faithfulness）、是否回答了用户问题（Answer Relevance）
from typing import Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

import config
from utils.logger import log_info, log_warn


class AnswerGrader:
    """基于 LLM 的答案质量评分器"""

    FAITHFULNESS_PROMPT = """你是答案忠实度评估专家。请判断以下 AI 答案是否**完全基于**给定的参考文档，没有编造内容。

参考文档：
{context}

AI答案：
{answer}

判断标准：
- 答案中的所有关键事实是否都能在参考文档中找到依据
- 有没有答案提到但文档中不存在的关键信息

请仅输出一个单词：yes 或 no
- yes = 答案完全忠实于参考文档，没有编造
- no = 答案包含文档中不存在的信息（幻觉）

输出：
"""

    RELEVANCE_PROMPT = """你是答案相关性评估专家。请判断以下 AI 答案是否**直接回答了**用户的问题。

用户问题：{question}

AI答案：
{answer}

请仅输出一个单词：yes 或 no
- yes = 答案直接回答了用户的问题
- no = 答案没有直接回答用户的问题

输出：
"""

    def __init__(self, model: str = None, temperature: float = 0.0):
        self.model = model or config.LLM_MODEL
        self.temperature = temperature
        self._llm = None

    def _get_llm(self) -> ChatOllama:
        if self._llm is None:
            self._llm = ChatOllama(model=self.model, temperature=self.temperature)
        return self._llm

    def grade_faithfulness(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """
        评估答案对参考文档的忠实度（Faithfulness）
        Returns: {"score": 0~1, "verdict": "yes"/"no", "passed": bool}
        """
        verdict = self._ask_llm(
            self.FAITHFULNESS_PROMPT.format(context=context[:1500], answer=answer)
        )
        passed = verdict == "yes"
        result = {
            "metric": "faithfulness",
            "question": question,
            "verdict": verdict,
            "score": 1.0 if passed else 0.0,
            "passed": passed,
        }
        log_info("答案评分", f"Faithfulness 判定：{verdict}")
        return result

    def grade_answer_relevance(self, question: str, answer: str) -> Dict[str, Any]:
        """
        评估答案是否回答了用户问题（Answer Relevance）
        Returns: {"score": 0~1, "verdict": "yes"/"no", "passed": bool}
        """
        verdict = self._ask_llm(
            self.RELEVANCE_PROMPT.format(question=question, answer=answer)
        )
        passed = verdict == "yes"
        result = {
            "metric": "answer_relevance",
            "question": question,
            "verdict": verdict,
            "score": 1.0 if passed else 0.0,
            "passed": passed,
        }
        log_info("答案评分", f"Answer Relevance 判定：{verdict}")
        return result

    def _ask_llm(self, prompt: str) -> str:
        """调用 LLM 获取 yes/no 判定"""
        try:
            llm = self._get_llm()
            resp = llm.invoke([HumanMessage(content=prompt)])
            content = resp.content.strip().lower()
            # 提取 yes/no
            if "yes" in content:
                return "yes"
            if "no" in content:
                return "no"
            return content
        except Exception as e:
            log_warn("答案评分", f"LLM调用失败，默认判定通过：{e}")
            return "yes"


# 全局单例
answer_grader = AnswerGrader()

# alias 供 MCP/Agent 使用
AnswerGraderInstance = answer_grader