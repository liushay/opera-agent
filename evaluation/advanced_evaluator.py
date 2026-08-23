# evaluation/advanced_evaluator.py 高级评测器
# 功能：在基础 Recall/HitRate/MRR/NDCG 之上，增加面向 Agentic RAG 的高级指标
#   1. Faithfulness（忠实度）：答案是否忠实于检索文档
#   2. Answer Relevance（答案相关性）：答案是否直接回答了问题
#   3. Context Precision（上下文精确率）：排序越靠前的文档越相关
#   4. Context Recall（上下文召回率）：相关文档是否都被召回
#   5. Agent Task Success（任务成功率）：Agent 端到端任务是否成功完成
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

from rag.agentic.answer_grader import answer_grader
from rag.agentic.document_grader import document_grader
from utils.logger import log_info, log_error
from utils.rag_exceptions import BaseRAGException


class AdvancedEvaluator:
    """高级 RAG/Agent 评测器"""

    def __init__(self):
        self.report_dir = "./evaluation_report"

    # ===================== 单指标计算 =====================

    def faithfulness(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """Faithfulness：答案忠实度"""
        return answer_grader.grade_faithfulness(question, answer, context)

    def answer_relevance(self, question: str, answer: str) -> Dict[str, Any]:
        """Answer Relevance：答案相关性"""
        return answer_grader.grade_answer_relevance(question, answer)

    def context_precision(
        self, query: str, docs: List[Document], relevant_indices: List[int]
    ) -> float:
        """
        Context Precision@K：相关文档在排序中越靠前，分数越高
        计算方式：对每个相关文档位置，累加 (相关文档数@位置 / 位置)，除以总相关文档数
        """
        if not relevant_indices:
            return 0.0

        total_precision = 0.0
        for k in relevant_indices:
            # 前 k 个位置中相关文档数
            relevant_in_top_k = sum(1 for i in relevant_indices if i <= k)
            total_precision += relevant_in_top_k / k

        return total_precision / len(relevant_indices)

    def context_recall(
        self, query: str, docs: List[Document], relevant_indices: List[int], total_relevant: int
    ) -> float:
        """
        Context Recall：召回到的相关文档数 / 总相关文档数
        relevant_indices: 检索结果中被判定相关的索引（1-based）
        total_relevant: 期望的相关文档总数
        """
        if total_relevant <= 0:
            return 0.0
        return len(relevant_indices) / total_relevant

    def agent_task_success(self, expected: bool, actual: bool) -> float:
        """Agent Task Success：任务是否成功完成（1=成功，0=失败）"""
        return 1.0 if expected == actual else 0.0

    # ===================== 端到端 Agent 评测 =====================

    def evaluate_agent_response(
        self,
        question: str,
        answer: str,
        context_docs: List[Document],
        graded_docs: List[Dict] = None,
        task_expected_success: bool = True,
        task_actual_success: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        对一次完整 Agent 响应进行高级评测
        Args:
            question: 用户问题
            answer: Agent 生成的答案
            context_docs: 检索到的上下文文档
            graded_docs: document_grader 评分结果（可选，若传入则计算 Context Precision/Recall）
            task_expected_success: 期望任务成功（默认 True）
            task_actual_success: 实际任务是否成功（若 None 则根据 answer 是否非空判断）
        Returns:
            包含所有高级指标的评测结果
        """
        context_text = "\n".join(d.page_content for d in context_docs)

        result = {
            "question": question,
            "record_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 1. Faithfulness
        result["faithfulness"] = self.faithfulness(question, answer, context_text)

        # 2. Answer Relevance
        result["answer_relevance"] = self.answer_relevance(question, answer)

        # 3. Context Precision / Recall
        if graded_docs is not None:
            relevant_indices = [
                i + 1 for i, g in enumerate(graded_docs) if g["relevant"]
            ]
            result["context_precision"] = round(
                self.context_precision(question, context_docs, relevant_indices), 4
            )
            total_relevant = max(len(relevant_indices), 1)  # 简化：以命中数近似
            result["context_recall"] = round(
                self.context_recall(question, context_docs, relevant_indices, total_relevant), 4
            )
        else:
            result["context_precision"] = None
            result["context_recall"] = None

        # 4. Agent Task Success
        if task_actual_success is None:
            task_actual_success = bool(answer and answer.strip())
        result["agent_task_success"] = self.agent_task_success(
            task_expected_success, task_actual_success
        )

        return result

    # ===================== 批量评测 =====================

    def run_batch_evaluation(
        self,
        cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        批量执行 Agent 响应高级评测
        cases: [{"question", "answer", "context_docs", "graded_docs"}]
        """
        results = []
        for case in cases:
            try:
                r = self.evaluate_agent_response(
                    question=case["question"],
                    answer=case.get("answer", ""),
                    context_docs=case.get("context_docs", []),
                    graded_docs=case.get("graded_docs"),
                )
                results.append(r)
            except Exception as e:
                log_error("高级评测", f"单条评测失败：{e}", e)

        # 汇总指标
        agg = {
            "faithfulness": self._avg_result(results, "faithfulness"),
            "answer_relevance": self._avg_result(results, "answer_relevance"),
            "context_precision": self._avg_result(results, "context_precision"),
            "context_recall": self._avg_result(results, "context_recall"),
            "agent_task_success": self._avg_result(results, "agent_task_success"),
        }

        summary = {
            "evaluate_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_cases": len(results),
            "overall_metrics": agg,
            "per_case_results": results,
        }

        # 保存报告
        report_path = os.path.join(self.report_dir, "advanced_eval_report.md")
        self._save_report(report_path, summary)

        return summary

    def _avg_result(self, results: List[Dict], key: str) -> Optional[float]:
        """计算指定指标的平均分"""
        vals = [
            r[key]["score"]
            if isinstance(r.get(key), dict) and "score" in r[key]
            else r.get(key)
            for r in results
        ]
        vals = [v for v in vals if v is not None and isinstance(v, (int, float))]
        if not vals:
            return None
        return round(sum(vals) / len(vals), 4)

    # ===================== 报告生成 =====================

    def _save_report(self, path: str, summary: Dict[str, Any]):
        """生成高级评测 Markdown 报告"""
        lines = []
        lines.append("# RAG / Agent 高级评测报告")
        lines.append("")
        lines.append(f"- **评测时间**：{summary['evaluate_time']}")
        lines.append(f"- **测试用例数**：{summary['total_cases']}")
        lines.append("")
        lines.append("## 整体平均指标")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        for k, v in summary["overall_metrics"].items():
            lines.append(f"| **{k}** | {v if v is not None else 'N/A'} |")
        lines.append("")
        lines.append("## 各用例详细结果")
        lines.append("")
        for i, r in enumerate(summary["per_case_results"], 1):
            lines.append(f"### 用例{i}：{r['question']}")
            lines.append("")
            lines.append("| 指标 | 判定 | 分数 |")
            lines.append("|------|------|------|")
            for k in ["faithfulness", "answer_relevance", "context_precision", "context_recall", "agent_task_success"]:
                val = r.get(k)
                if val is None:
                    continue
                if isinstance(val, dict):
                    judge = val.get("verdict", "N/A")
                    score = val.get("score", "N/A")
                else:
                    judge = "N/A"
                    score = val
                lines.append(f"| {k} | {judge} | {score} |")
            lines.append("")
            lines.append("---")
            lines.append("")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        log_info("高级评测", f"报告已保存：{path}")


# 全局单例
advanced_evaluator = AdvancedEvaluator()