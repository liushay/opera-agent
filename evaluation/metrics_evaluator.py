# evaluation/metrics_evaluator.py 检索指标评测器
# 功能：
#   1. 定义评测问题集（query -> 期望命中关键词）
#   2. 执行混合检索，收集召回文档
#   3. 计算召回率(Recall@K)、命中率(HitRate@K)、MRR@K、NDCG@K
#   4. 多次运行取均值，结果保存至单独评测报告文档
import os
import json
from datetime import datetime
from typing import List, Dict, Any

import config
from rag.vectorstore import hybrid_retrieve
from utils.logger import log_info, log_warn, log_error
from utils.rag_exceptions import BaseRAGException
from evaluation.dynamic_questions import generate_dynamic_question_set


class MetricsEvaluator:
    """RAG检索指标量化评测器"""

    def __init__(self):
        self.report_dir = os.path.dirname(config.EVAL_REPORT_PATH)
        os.makedirs(self.report_dir, exist_ok=True)

    # ===================== 指标计算方法 =====================

    def _hit_rate_at_k(self, relevant_in_results: List[bool], k: int) -> float:
        """
        命中率 HitRate@K：前K条结果中是否包含至少一条相关文档
        relevant_in_results: 每个位置是否相关(True/False)的列表
        """
        if k <= 0:
            return 0.0
        top_k = relevant_in_results[:k]
        return 1.0 if any(top_k) else 0.0

    def _recall_at_k(self, relevant_in_results: List[bool], total_relevant: int, k: int) -> float:
        """
        召回率 Recall@K：前K条结果中相关文档数 / 总相关文档数
        relevant_in_results: 每个位置是否相关的列表
        total_relevant: 该query期望命中的总相关文档数
        """
        if total_relevant <= 0 or k <= 0:
            return 0.0
        top_k_relevant = sum(relevant_in_results[:k])
        return top_k_relevant / total_relevant

    def _mrr_at_k(self, relevant_in_results: List[bool], k: int) -> float:
        """
        平均倒数排名 MRR@K：第一个相关文档排名的倒数
        """
        if k <= 0:
            return 0.0
        for idx, is_rel in enumerate(relevant_in_results[:k]):
            if is_rel:
                return 1.0 / (idx + 1)  # idx为0时rank=1
        return 0.0

    def _ndcg_at_k(self, relevant_in_results: List[bool], k: int) -> float:
        """
        归一化折损累计增益 NDCG@K
        简化实现：相关性为二元(0/1)，DCG = sum(rel_i / log2(i+2))
        """
        if k <= 0:
            return 0.0

        # 计算理想DCG（所有相关文档排在前面）
        total_relevant = sum(relevant_in_results)
        ideal_dcg = 0.0
        for i in range(min(total_relevant, k)):
            ideal_dcg += 1.0 / (i + 2)  # log2(i+2) 近似为 i+2 简化，保持单调

        if ideal_dcg == 0.0:
            return 0.0

        # 实际DCG
        dcg = 0.0
        for i, is_rel in enumerate(relevant_in_results[:k]):
            if is_rel:
                dcg += 1.0 / (i + 2)

        return dcg / ideal_dcg

    # ===================== 相关性判定 =====================

    def _judge_relevance(
        self,
        retrieved_docs: List[Any],
        expected_keywords: List[str],
    ) -> List[bool]:
        """
        判定检索结果的每一条文档是否与问题相关
        判定规则：文档正文中是否包含期望关键词中的任意一个
        """
        relevant_flags = []
        for doc in retrieved_docs:
            text = doc.page_content
            # 关键词命中：期望关键词中至少一个出现在文档文本中
            hit = any(kw.lower() in text.lower() for kw in expected_keywords if kw)
            relevant_flags.append(hit)
        return relevant_flags

    # ===================== 单条查询评测 =====================

    def _evaluate_single_query(
        self, query: str, expected_keywords: List[str]
    ) -> Dict[str, Any]:
        """
        评测单条查询的检索效果
        返回各K值的指标及检索结果详情
        若检索执行失败（Ollama断开等），返回错误标记而非抛出异常，不中断整体评测
        """
        try:
            # 执行混合检索
            docs = hybrid_retrieve(query)
        except Exception as e:
            log_warn("指标评测", f"查询 [{query[:30]}...] 检索执行失败，跳过：{e}")
            return {
                "query": query,
                "expected_keywords": expected_keywords,
                "retrieved_count": 0,
                "retrieved_texts": [],
                "relevant_flags": [],
                "metrics": {},
                "error": str(e),
            }

        retrieved_texts = [d.page_content for d in docs]

        # 判定相关性
        relevant_flags = self._judge_relevance(docs, expected_keywords)
        # 期望相关文档数：关键词命中数（至少1，避免除零）
        total_relevant = max(1, min(len(expected_keywords), len(retrieved_texts)))  # 期望相关文档数

        # 计算各K值指标
        metrics = {}
        for k in config.EVAL_TOP_K_LIST:
            metrics[f"recall@{k}"] = round(
                self._recall_at_k(relevant_flags, total_relevant, k), 4
            )
            metrics[f"hit_rate@{k}"] = round(
                self._hit_rate_at_k(relevant_flags, k), 4
            )
            metrics[f"mrr@{k}"] = round(
                self._mrr_at_k(relevant_flags, k), 4
            )
            metrics[f"ndcg@{k}"] = round(
                self._ndcg_at_k(relevant_flags, k), 4
            )

        return {
            "query": query,
            "expected_keywords": expected_keywords,
            "retrieved_count": len(docs),
            "retrieved_texts": retrieved_texts,
            "relevant_flags": relevant_flags,
            "metrics": metrics,
        }

    # ===================== 整体评测入口 =====================

    def run_evaluation(
        self,
        rounds: int = None,
        topic: str = "",
        reference_text: str = "",
        question_count: int = None,
    ) -> Dict[str, Any]:
        """
        执行完整评测（改造：问题来源支持动态生成）：
        1. 获取测评问题集：
           - 当传入 topic（或 reference_text）时：动态生成测评问题集（不再读取config固定5题）
           - 未传 topic 且 config 中仍配置了问题集时：兼容旧调用，使用 config.EVAL_QUESTION_SET
           - 均无问题时：自动以默认主题从知识库动态生成
        2. 对每个问题执行检索并计算指标（原指标计算逻辑不变）
        3. 对多条测试问题计算平均指标
        4. 多次运行取均值（降低随机性影响）
        5. 生成Markdown评测报告单独保存
        """
        rounds = rounds or config.EVAL_ROUNDS
        question_set = list(config.EVAL_QUESTION_SET or [])
        # ===== 动态测评：不再依赖config固定问题集 =====
        if topic or reference_text or not question_set:
            dynamic_topic = topic or "京剧艺术"
            log_info("指标评测", f"使用动态测评问题生成器，主题：{dynamic_topic}")
            question_set = generate_dynamic_question_set(
                topic=dynamic_topic,
                reference_text=reference_text,
                question_count=question_count,
            )
        log_info("指标评测", f"开始RAG检索指标评测，共{len(question_set)}道测试题，运行{rounds}轮")
        # 兼容动态生成问题中的 question_type 字段（extra信息，不影响指标计算）
        question_meta = {}
        for q in question_set:
            question_meta[q["query"]] = {
                "question_type": q.get("question_type", "unknown"),
            }

        # 累计所有轮次各指标
        agg_metrics: Dict[str, float] = {}
        per_query_detail = {q["query"]: [] for q in question_set}
        failed_queries: Dict[str, int] = {}  # 记录各查询失败轮次数

        try:
            for round_idx in range(1, rounds + 1):
                log_info("指标评测", f"===== 第 {round_idx}/{rounds} 轮评测开始 =====")
                for q_item in question_set:
                    query = q_item["query"]
                    keywords = q_item["expected_keywords"]
                    result = self._evaluate_single_query(query, keywords)
                    if result.get("error"):
                        failed_queries[query] = failed_queries.get(query, 0) + 1
                        log_warn("指标评测", f"查询 [{query[:30]}...] 第{round_idx}轮检索失败：{result['error']}")
                    # 累计指标（失败的查询 metrics 为空，不影响累计）
                    for k, v in result["metrics"].items():
                        agg_metrics[k] = agg_metrics.get(k, 0.0) + v
                    per_query_detail[query].append(result)
                log_info("指标评测", f"第 {round_idx} 轮评测完成")

            # 计算有效评测次数（排除失败的查询-轮次，至少为1避免除零）
            total_evals = max(1, rounds * len(question_set) - sum(failed_queries.values()))
            # 计算最终平均指标（多轮取均值，失败查询不计入分母）
            final_metrics = {}
            for k, total in agg_metrics.items():
                final_metrics[k] = round(total / total_evals, 4)

            summary = {
                "evaluate_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rounds": rounds,
                "question_count": len(question_set),
                "top_k_list": config.EVAL_TOP_K_LIST,
                "final_metrics": final_metrics,
                "per_query_details": {
                    q: {"avg_metrics": self._avg_query_metrics(details), "rounds_data": details}
                    for q, details in per_query_detail.items()
                },
                # 动态生成问题的类型元数据（兼容旧评测：question_set无类型时为空）
                "question_meta": question_meta,
            }

            # 生成并保存评测报告
            report_path = self._generate_report(summary)
            summary["report_path"] = report_path

            log_info("指标评测", f"评测完成，报告已保存至：{report_path}")
            return summary

        except BaseRAGException as e:
            err_msg = f"评测执行失败：{e.msg}"
            log_error("指标评测失败", err_msg, e.origin_err)
            raise
        except Exception as e:
            err_msg = "评测执行出现未知错误"
            log_error("指标评测异常", err_msg, e)
            raise

    def _avg_query_metrics(self, round_results: List[Dict]) -> Dict[str, float]:
        """对单条查询多轮结果计算平均指标（排除失败轮次）"""
        if not round_results:
            return {}
        # 收集所有有效轮次的指标
        valid_results = [r for r in round_results if not r.get("error")]
        if not valid_results:
            return {}
        avg = {}
        first_metrics = valid_results[0]["metrics"]
        valid_count = len(valid_results)
        for k in first_metrics:
            avg[k] = round(
                sum(r["metrics"][k] for r in valid_results) / valid_count, 4
            )
        return avg

    # ===================== 报告生成 =====================

    def _generate_report(self, summary: Dict[str, Any]) -> str:
        """生成Markdown格式评测报告并保存到独立文件"""
        lines = []
        lines.append("# RAG 检索指标评测报告")
        lines.append("")
        lines.append(f"- **评测时间**：{summary['evaluate_time']}")
        lines.append(f"- **评测轮次**：{summary['rounds']} 轮")
        lines.append(f"- **测试问题数**：{summary['question_count']} 道")
        lines.append(f"- **评测K值**：{summary['top_k_list']}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # 整体指标表
        lines.append("## 整体平均指标")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        fm = summary["final_metrics"]
        for k, v in fm.items():
            lines.append(f"| **{k}** | {v} |")
        lines.append("")
        lines.append("---")
        lines.append("")

        # 各查询详情
        lines.append("## 各问题详细指标（均值）")
        lines.append("")
        for query, detail in summary["per_query_details"].items():
            lines.append(f"### 问题：{query}")
            lines.append("")
            lines.append("| 指标 | 数值 |")
            lines.append("|------|------|")
            for k, v in detail["avg_metrics"].items():
                lines.append(f"| {k} | {v} |")
            lines.append("")
            # 展示第一轮的检索结果详情
            first_round = detail["rounds_data"][0]
            lines.append(f"**期望关键词**：{', '.join(first_round['expected_keywords'])}")
            lines.append("")
            if first_round.get("error"):
                lines.append(f"⚠️ **检索失败**：{first_round['error']}")
                lines.append("")
            else:
                lines.append("**检索结果命中情况**：")
                lines.append("")
                for i, (text, is_rel) in enumerate(
                    zip(first_round["retrieved_texts"], first_round["relevant_flags"]), 1
                ):
                    mark = "✅ 命中" if is_rel else "❌ 未命中"
                    preview = text[:60].replace("\n", " ")
                    lines.append(f"{i}. [{mark}] {preview}...")
                lines.append("")
            lines.append("---")
            lines.append("")

        # 附录：原始测试数据
        lines.append("## 附录：原始测试数据")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        lines.append("```")
        lines.append("")

        # 写入文件
        report_path = config.EVAL_REPORT_PATH
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            log_info("评测报告", f"报告已写入：{report_path}")
        except Exception as e:
            log_error("评测报告写入失败", f"无法写入报告：{report_path}", e)
            raise

        return report_path


# 全局单例
metrics_evaluator = MetricsEvaluator()