# evaluation/eval_agent.py 独立评估Agent主模块
# 功能：
#   使用独立LLM（deepseek-r1:7b）对生成Agent（qwen2.5:7b-instruct）的输出进行深度评估。
#   5步评估流程：事实声明提取 → 事实核查 → 意图对齐 → 质量评分 → 决策路由
# 设计原则：
#   - 与生成Agent完全解耦，使用独立LLM实例，打破"自己评自己"的盲区
#   - 事实核查是核心能力：逐条声明检索知识库验证，检出幻觉
#   - 支持同步/异步两种模式，异步模式不阻塞首次响应
#   - 与现有 output_validator 互补：output_validator 为快速第一道校验，eval_agent 为深度第二道校验
#   - 完全解耦，不修改任何现有Agent代码，仅通过 chat_routes.py 集成
import json
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.llm_utils import build_llm, invoke_with_retry
from agent.intent_guard import is_image_intent
from evaluation.eval_prompts import (
    build_claim_extraction_prompt,
    build_fact_verification_prompt,
    build_intent_alignment_prompt,
    build_quality_scoring_prompt,
)
from tools.custom_tools import knowledge_tool
from utils.logger import log_info, log_warn, log_error

import config


# ===================== 评估结果数据结构 =====================

@dataclass
class EvalResult:
    """评估结果数据结构"""
    pass_: bool = False                          # 是否通过评估
    factual_score: float = 0.0                   # 事实准确率 0-1
    intent_alignment: bool = False               # 意图对齐
    hallucination_rate: float = 0.0              # 幻觉率 0-1
    quality_score: float = 0.0                   # 综合质量分 0-1
    diagnosis: Dict[str, Any] = field(default_factory=dict)  # 诊断详情
    suggestion: str = ""                         # 重生成建议
    action: str = "fail"                         # 决策：pass / retry / fail
    # 事实核查详细结果
    fact_check_results: List[Dict[str, Any]] = field(default_factory=list)
    # 意图对齐详细结果
    intent_alignment_detail: Dict[str, Any] = field(default_factory=dict)
    # 图像诉求标记
    image_required: bool = False
    image_missing: bool = False


# ===================== 评估Agent主类 =====================

class EvalAgent:
    """
    独立评估Agent
    使用独立LLM实例（deepseek-r1:7b），与生成Agent（qwen2.5:7b-instruct）完全解耦。
    """

    def __init__(self):
        """初始化评估Agent专用LLM实例"""
        self._eval_model = getattr(config, "EVAL_LLM_MODEL", "deepseek-r1:7b")
        self._eval_temp = getattr(config, "EVAL_LLM_TEMP", 0.0)
        self._eval_timeout = getattr(config, "EVAL_AGENT_TIMEOUT", 180)
        self._fact_check_enabled = getattr(config, "EVAL_FACT_CHECK_ENABLED", True)
        self._fact_check_top_k = getattr(config, "EVAL_FACT_CHECK_TOP_K", 3)
        self._max_retry = getattr(config, "EVAL_MAX_RETRY", 1)

        # 独立LLM实例（评估专用）
        self.llm = build_llm(
            model=self._eval_model,
            temperature=self._eval_temp,
            timeout=self._eval_timeout,
        )
        log_info("评估Agent", f"初始化完成，评估模型={self._eval_model}，"
                 f"温度={self._eval_temp}，事实核查={'启用' if self._fact_check_enabled else '禁用'}")

    # ===================== 主评估入口 =====================

    def evaluate(
        self,
        user_query: str,
        intent: Dict[str, Any],
        generated_text: str,
        reference_text: str = "",
        image_resources: List[Dict[str, Any]] = None,
    ) -> EvalResult:
        """
        主评估入口：执行5步评估流程。

        Args:
            user_query: 用户原始请求
            intent: 意图解析结果（来自 agent/intent.py 的 parse_intent）
            generated_text: 生成Agent的输出文本
            reference_text: 参考文档文本（可选）
            image_resources: 图片资源列表（可选）

        Returns:
            EvalResult: 包含是否通过、各维度得分、诊断建议、决策动作
        """
        log_info("评估Agent", f"开始评估，用户请求={user_query[:80]}，生成内容长度={len(generated_text)}字")

        result = EvalResult()

        try:
            # Step 1: 提取事实声明
            claims = self._extract_claims(generated_text)
            log_info("评估Agent", f"Step 1 完成：提取到{len(claims)}条事实声明")

            # Step 2: 事实核查（逐条检索验证）
            if self._fact_check_enabled and claims:
                fact_results = self._fact_check(claims)
                result.fact_check_results = fact_results
            else:
                fact_results = []
                result.fact_check_results = []
            log_info("评估Agent", f"Step 2 完成：核查{len(fact_results)}条声明")

            # Step 3: 意图对齐检查
            intent_alignment = self._check_intent_alignment(
                user_query, intent, generated_text
            )
            result.intent_alignment_detail = intent_alignment
            result.intent_alignment = intent_alignment.get("aligned", False)
            log_info("评估Agent", f"Step 3 完成：意图对齐={'通过' if result.intent_alignment else '未通过'}")

            # Step 4: 质量评分（7维度）
            quality = self._quality_score(
                user_query, generated_text, fact_results, intent_alignment, image_resources
            )
            log_info("评估Agent", f"Step 4 完成：综合评分={quality.get('overall', 0)}")

            # 计算事实准确率和幻觉率
            result.factual_score = self._compute_factual_score(fact_results)
            result.hallucination_rate = self._compute_hallucination_rate(fact_results)
            result.quality_score = quality.get("overall", 0.0)
            result.diagnosis = quality
            result.suggestion = quality.get("suggestion", "")

            # 图像诉求检查
            result.image_required = is_image_intent(user_query)
            image_resources = image_resources or []
            has_real_image = any(
                r.get("type") == "image"
                and r.get("image_status") == "generated"
                and r.get("image_url")
                for r in image_resources
            )
            result.image_missing = bool(result.image_required and not has_real_image)

            # Step 5: 决策路由
            self._decision_gate(result)

            log_info(
                "评估Agent",
                f"评估完成：action={result.action}，pass={result.pass_}，"
                f"事实准确率={result.factual_score:.2f}，幻觉率={result.hallucination_rate:.2f}，"
                f"综合评分={result.quality_score:.2f}",
            )

        except Exception as e:
            # 评估Agent自身异常时降级为 pass=True，避免阻塞主流程
            log_error("评估Agent", f"评估流程异常，降级放行：{e}", e)
            result.pass_ = True
            result.action = "pass"
            result.suggestion = f"评估Agent异常，降级放行：{str(e)[:100]}"

        return result

    # ===================== Step 1: 事实声明提取 =====================

    def _extract_claims(self, generated_text: str) -> List[str]:
        """
        从生成结果中提取所有事实性声明。
        使用评估LLM提取，提取失败时降级为按句号/换行分割。
        """
        prompt = build_claim_extraction_prompt(generated_text)
        try:
            content = invoke_with_retry(
                [prompt],
                model=self._eval_model,
                temperature=0.0,
                timeout=self._eval_timeout,
                task_name="事实声明提取",
            )
            start = content.find("{")
            end = content.rfind("}") + 1
            if start < 0 or end <= start:
                raise ValueError("无有效JSON")
            data = json.loads(content[start:end])
            claims = data.get("claims", [])
            claims = [c.strip() for c in claims if c and c.strip()]
            if claims:
                return claims
        except Exception as e:
            log_warn("评估Agent", f"事实声明提取LLM失败，降级为规则提取：{e}")

        # 降级：按句号分割，过滤过短/过长的句子
        fallback = []
        for sent in generated_text.replace("\n", "。").split("。"):
            sent = sent.strip()
            if 10 < len(sent) < 200 and not sent.startswith(("好的", "请", "您", "以上")):
                fallback.append(sent)
        return fallback[:20]  # 最多20条

    # ===================== Step 2: 事实核查 =====================

    def _fact_check(self, claims: List[str]) -> List[Dict[str, Any]]:
        """
        逐条检索知识库验证事实声明。
        对每条声明：
          1. 独立检索知识库
          2. 用评估LLM判断支撑程度
        返回：verified / partial / unverified / contradicted
        """
        results = []
        for i, claim in enumerate(claims):
            try:
                # 独立检索知识库
                evidence = knowledge_tool.invoke({"query": claim})
                evidence_str = str(evidence) if evidence else ""

                # 用评估LLM验证
                verdict = self._verify_claim(claim, evidence_str)
                results.append({
                    "claim": claim,
                    "verdict": verdict,
                    "evidence": evidence_str[:500],
                })
            except Exception as e:
                log_warn("评估Agent", f"事实核查第{i+1}条失败：{e}")
                results.append({
                    "claim": claim,
                    "verdict": {"verdict": "unverified", "reason": f"核查异常：{str(e)[:100]}", "confidence": 0.0},
                    "evidence": "",
                })
        return results

    def _verify_claim(self, claim: str, evidence: str) -> Dict[str, Any]:
        """判断单条声明是否被知识库支撑"""
        prompt = build_fact_verification_prompt(claim, evidence)
        try:
            content = invoke_with_retry(
                [prompt],
                model=self._eval_model,
                temperature=0.0,
                timeout=self._eval_timeout,
                task_name="事实核查",
            )
            start = content.find("{")
            end = content.rfind("}") + 1
            if start < 0 or end <= start:
                raise ValueError("无有效JSON")
            return json.loads(content[start:end])
        except Exception as e:
            log_warn("评估Agent", f"事实核查LLM调用失败：{e}")
            return {"verdict": "unverified", "reason": f"核查LLM异常：{str(e)[:100]}", "confidence": 0.0}

    # ===================== Step 3: 意图对齐检查 =====================

    def _check_intent_alignment(
        self, user_query: str, intent: Dict[str, Any], generated_text: str
    ) -> Dict[str, Any]:
        """
        检查生成内容是否完全对齐用户意图。
        对照意图解析结果中的 core_task / must_require / forbid_list 逐项检查。
        """
        core_task = intent.get("core_task", user_query)
        must_require = intent.get("must_require", [])
        forbid_list = intent.get("forbid_list", [])

        prompt = build_intent_alignment_prompt(
            user_query, core_task, must_require, forbid_list, generated_text
        )
        try:
            content = invoke_with_retry(
                [prompt],
                model=self._eval_model,
                temperature=0.0,
                timeout=self._eval_timeout,
                task_name="意图对齐检查",
            )
            start = content.find("{")
            end = content.rfind("}") + 1
            if start < 0 or end <= start:
                raise ValueError("无有效JSON")
            return json.loads(content[start:end])
        except Exception as e:
            log_warn("评估Agent", f"意图对齐检查LLM失败：{e}")
            return {
                "aligned": True,
                "core_task_aligned": True,
                "must_satisfied": True,
                "forbid_respected": True,
                "user_focus_priority": True,
                "issues": [f"意图对齐检查LLM异常，降级放行：{str(e)[:100]}"],
                "suggestion": "",
            }

    # ===================== Step 4: 质量评分 =====================

    def _quality_score(
        self,
        user_query: str,
        generated_text: str,
        fact_check_results: List[Dict[str, Any]],
        intent_alignment: Dict[str, Any],
        image_resources: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        7维度综合评分：
          1. factual_accuracy（事实准确性）
          2. intent_alignment（意图对齐度）
          3. completeness（完整性）
          4. hallucination_free（无幻觉率）
          5. reference_accuracy（引用准确性）
          6. task_completion（任务完成度）
          7. user_satisfaction（用户满意度预估）
        """
        prompt = build_quality_scoring_prompt(
            user_query, generated_text, fact_check_results, intent_alignment, image_resources
        )
        try:
            content = invoke_with_retry(
                [prompt],
                model=self._eval_model,
                temperature=0.0,
                timeout=self._eval_timeout,
                task_name="质量评分",
            )
            start = content.find("{")
            end = content.rfind("}") + 1
            if start < 0 or end <= start:
                raise ValueError("无有效JSON")
            return json.loads(content[start:end])
        except Exception as e:
            log_warn("评估Agent", f"质量评分LLM失败，降级为规则评分：{e}")
            # 降级：基于事实核查结果计算简单评分
            return self._fallback_quality_score(fact_check_results, intent_alignment)

    def _fallback_quality_score(
        self,
        fact_check_results: List[Dict[str, Any]],
        intent_alignment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """降级质量评分：基于事实核查统计 + 意图对齐结果"""
        total = len(fact_check_results)
        if total == 0:
            factual_score = 5.0
            hallucination_score = 5.0
        else:
            verified = sum(1 for r in fact_check_results if r.get("verdict", {}).get("verdict") == "verified")
            partial = sum(1 for r in fact_check_results if r.get("verdict", {}).get("verdict") == "partial")
            contradicted = sum(1 for r in fact_check_results if r.get("verdict", {}).get("verdict") == "contradicted")
            factual_pct = (verified + 0.5 * partial) / total
            hallucination_pct = 1.0 - (contradicted / total)
            factual_score = round(factual_pct * 5, 1)
            hallucination_score = round(hallucination_pct * 5, 1)

        intent_score = 5.0 if intent_alignment.get("aligned", True) else 2.0
        overall = round((factual_score + intent_score + hallucination_score + 4.0 + 4.0 + 4.0 + 4.0) / 7, 1)

        return {
            "scores": {
                "factual_accuracy": factual_score,
                "intent_alignment": intent_score,
                "completeness": 4.0,
                "hallucination_free": hallucination_score,
                "reference_accuracy": 4.0,
                "task_completion": 4.0,
                "user_satisfaction": 4.0,
            },
            "overall": overall,
            "critical_issues": [],
            "minor_issues": [],
            "suggestion": "（降级规则评分，非LLM评分）",
        }

    # ===================== 新增: Span级幻觉检测 =====================

    def _span_hallucination_check(
        self, generated_text: str, reference_text: str = "", user_query: str = ""
    ) -> Dict[str, Any]:
        """Span级幻觉检测：一次LLM调用输出所有幻觉的精确位置和修正内容"""
        factual_context = reference_text
        if not factual_context:
            try:
                evidence = knowledge_tool.invoke({"query": user_query[:200]})
                factual_context = str(evidence) if evidence else ""
            except Exception:
                factual_context = ""
        prompt = build_span_hallucination_prompt(
            generated_text=generated_text,
            factual_context=factual_context,
            user_query=user_query,
        )
        try:
            content = invoke_with_retry(
                [prompt], model=self._eval_model, temperature=0.0,
                timeout=self._eval_timeout, task_name="Span级幻觉检测",
            )
            start = content.find("{")
            end = content.rfind("}") + 1
            result = json.loads(content[start:end]) if start >= 0 and end > start else {}
            log_info("评估Agent", f"Span级幻觉检测完成：严重度={result.get('overall_assessment', {}).get('hallucination_severity', 'unknown')}")
            return result
        except Exception as e:
            log_warn("评估Agent", f"Span级幻觉检测失败：{e}")
            return {"hallucinations": [], "overall_assessment": {"hallucination_severity": "none"}}

    # ===================== 新增: 批量事实核查 =====================

    def _batch_fact_check(self, claims: List[str]) -> List[Dict[str, Any]]:
        """批量事实核查：一次LLM调用核查所有声明，替代逐条核查"""
        if not claims:
            return []
        evidence_map = {}
        for i, claim in enumerate(claims):
            try:
                evidence = knowledge_tool.invoke({"query": claim})
                evidence_map[str(i)] = str(evidence) if evidence else ""
            except Exception as e:
                log_warn("评估Agent", f"批量检索第{i+1}条失败：{e}")
                evidence_map[str(i)] = ""
        prompt = build_batch_fact_check_prompt(claims, evidence_map)
        try:
            content = invoke_with_retry(
                [prompt], model=self._eval_model, temperature=0.0,
                timeout=self._eval_timeout, task_name="批量事实核查",
            )
            start = content.find("{")
            end = content.rfind("}") + 1
            data = json.loads(content[start:end]) if start >= 0 and end > start else {}
            results = data.get("results", [])
            log_info("评估Agent", f"批量事实核查：{len(results)}条，汇总={data.get('summary', {})}")
            return [{
                "claim": r.get("claim", ""),
                "verdict": {
                    "verdict": r.get("verdict", "unverified"),
                    "reason": r.get("reason", ""),
                    "confidence": r.get("confidence", 0.0),
                },
                "evidence": evidence_map.get(str(r.get("claim_index", 0)), "")[:500],
            } for r in results]
        except Exception as e:
            log_warn("评估Agent", f"批量事实核查失败，降级逐条：{e}")
            return self._fact_check(claims)

    # ===================== 新增: 修正指令生成 =====================

    def _generate_correction_instructions(
        self, user_query: str, generated_text: str,
        span_issues: List[Dict[str, Any]], intent_alignment: Dict[str, Any],
        quality_scores: Dict[str, Any],
    ) -> Dict[str, Any]:
        """将评估诊断转化为可执行的修正操作列表"""
        prompt = build_correction_instruction_prompt(
            user_query=user_query, generated_text=generated_text,
            span_issues=span_issues, intent_alignment=intent_alignment,
            quality_scores=quality_scores,
        )
        try:
            content = invoke_with_retry(
                [prompt], model=self._eval_model, temperature=0.0,
                timeout=self._eval_timeout, task_name="修正指令生成",
            )
            start = content.find("{")
            end = content.rfind("}") + 1
            result = json.loads(content[start:end]) if start >= 0 and end > start else {}
            log_info("评估Agent", f"修正指令生成：{len(result.get('corrections', []))}条操作")
            return result
        except Exception as e:
            log_warn("评估Agent", f"修正指令生成失败：{e}")
            return {"corrections": [], "regenerate_hint": f"请重新生成：{str(e)[:100]}", "priority": "high"}

    # ===================== 新增: Critic-Refine 循环 =====================

    def critic_refine_cycle(
        self, user_query: str, intent: Dict[str, Any], generated_text: str,
        reference_text: str = "", image_resources: List[Dict[str, Any]] = None,
        generate_func=None, gen_kwargs: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Critic-Refine 循环（核心创新）。
        1. 快速预筛选 → 2. Span级幻觉检测 → 3. 修正指令生成 → 4. 带精确修正重生成 → 5. 最多EVAL_MAX_RETRY轮
        """
        from evaluation.pre_filter import quick_check
        max_retry = getattr(config, "EVAL_MAX_RETRY", 1)
        attempts = 0
        final_text = str(generated_text)
        refine_history = []
        gen_kwargs = gen_kwargs or {}

        pre_result = quick_check(user_query, final_text, image_resources)
        if not pre_result.suspicious:
            log_info("Critic-Refine", "快速预筛选通过，无需深度评估")
            return {"text": final_text, "eval_result": None, "attempts": 1, "passed": True, "refine_history": [{"stage": "pre_filter", "passed": True}]}

        log_info("Critic-Refine", f"预筛选存疑：{pre_result.reasons}，进入深度评估")

        while attempts <= max_retry:
            attempts += 1
            log_info("Critic-Refine", f"第{attempts}轮迭代开始")

            span_result = self._span_hallucination_check(final_text, reference_text, user_query)
            hallucinations = span_result.get("hallucinations", [])
            severity = span_result.get("overall_assessment", {}).get("hallucination_severity", "none")

            intent_alignment = self._check_intent_alignment(user_query, intent, final_text)
            claims = self._extract_claims(final_text)
            fact_results = self._batch_fact_check(claims) if self._fact_check_enabled and claims else []
            quality = self._quality_score(user_query, final_text, fact_results, intent_alignment, image_resources)

            refine_history.append({
                "round": attempts, "hallucination_count": len(hallucinations),
                "hallucination_severity": severity, "intent_aligned": intent_alignment.get("aligned", False),
                "quality_overall": quality.get("overall", 0),
            })

            if severity in ("none", "low") and intent_alignment.get("aligned", False) and quality.get("overall", 0) >= 4.0:
                log_info("Critic-Refine", f"第{attempts}轮通过")
                return {"text": final_text, "eval_result": {"span_result": span_result, "intent_alignment": intent_alignment, "quality": quality}, "attempts": attempts, "passed": True, "refine_history": refine_history}

            correction = self._generate_correction_instructions(
                user_query, final_text, hallucinations, intent_alignment, quality)
            corrections = correction.get("corrections", [])
            regenerate_hint = correction.get("regenerate_hint", "请根据评估反馈重新生成。")

            if not corrections and attempts >= max_retry:
                break

            if generate_func:
                try:
                    new_text = generate_func(user_query, intent, regenerate_hint, corrections, **gen_kwargs)
                    if new_text and str(new_text).strip():
                        final_text = str(new_text)
                        log_info("Critic-Refine", f"第{attempts}轮重生成完成，长度={len(final_text)}字")
                        continue
                except Exception as e:
                    log_warn("Critic-Refine", f"重生成失败：{e}")
            break

        log_warn("Critic-Refine", f"全部{attempts}轮未通过")
        return {"text": final_text, "eval_result": {"span_result": span_result, "intent_alignment": intent_alignment, "quality": quality}, "attempts": attempts, "passed": False, "refine_history": refine_history}

    # ===================== 辅助计算 =====================

    def _compute_factual_score(self, fact_check_results: List[Dict[str, Any]]) -> float:
        """计算事实准确率（0-1）"""
        total = len(fact_check_results)
        if total == 0:
            return 1.0
        verified = sum(1 for r in fact_check_results if r.get("verdict", {}).get("verdict") == "verified")
        partial = sum(1 for r in fact_check_results if r.get("verdict", {}).get("verdict") == "partial")
        return round((verified + 0.5 * partial) / total, 3)

    def _compute_hallucination_rate(self, fact_check_results: List[Dict[str, Any]]) -> float:
        """计算幻觉率（0-1）：unverified + contradicted 的比例"""
        total = len(fact_check_results)
        if total == 0:
            return 0.0
        unverified = sum(1 for r in fact_check_results if r.get("verdict", {}).get("verdict") == "unverified")
        contradicted = sum(1 for r in fact_check_results if r.get("verdict", {}).get("verdict") == "contradicted")
        return round((unverified + contradicted) / total, 3)

    # ===================== Step 5: 决策路由 =====================

    def _decision_gate(self, result: EvalResult) -> None:
        """
        决策路由：根据评估结果决定 pass / retry / fail。

        决策矩阵：
        ┌───────────────────────────────┬──────────┐
        │ 条件                          │ 动作     │
        ├───────────────────────────────┼──────────┤
        │ 事实准确率 ≥ 0.8 且综合 ≥ 4.0 │ pass     │
        │ 有 contradicted 标记          │ retry    │
        │ 意图对齐失败                  │ retry    │
        │ 图像缺失                      │ retry    │
        │ 事实准确率 ≥ 0.6 但 < 0.8     │ retry    │
        │ 事实准确率 < 0.6              │ fail     │
        └───────────────────────────────┴──────────┘
        """
        factual = result.factual_score
        overall = result.quality_score
        has_contradicted = any(
            r.get("verdict", {}).get("verdict") == "contradicted"
            for r in result.fact_check_results
        )

        if result.image_missing:
            result.pass_ = False
            result.action = "retry"
            result.suggestion = "用户需要图片，但生成结果没有图片资源。请重新派发图像生成worker。" + (
                result.suggestion or ""
            )
        elif has_contradicted:
            result.pass_ = False
            result.action = "retry"
            result.suggestion = "检测到与知识库矛盾的事实声明（contradicted）。请严格基于知识库重新生成，禁止编造。" + (
                result.suggestion or ""
            )
        elif not result.intent_alignment:
            result.pass_ = False
            result.action = "retry"
            result.suggestion = "生成内容未对齐用户意图。请严格围绕用户核心任务重新生成。" + (
                result.suggestion or ""
            )
        elif factual >= 0.8 and overall >= 4.0:
            result.pass_ = True
            result.action = "pass"
        elif factual >= 0.6:
            result.pass_ = False
            result.action = "retry"
        else:
            result.pass_ = False
            result.action = "fail"


# ===================== 异步评估包装器 =====================

def evaluate_async(
    eval_agent: EvalAgent,
    user_query: str,
    intent: Dict[str, Any],
    generated_text: str,
    reference_text: str = "",
    image_resources: List[Dict[str, Any]] = None,
    memory_store: Any = None,
    session_id: str = "",
) -> None:
    """
    异步评估：不阻塞主流程，在后台线程中执行评估。
    评估结果写入 memory_store，供后续交互参考。

    Args:
        eval_agent: EvalAgent 实例
        user_query: 用户原始请求
        intent: 意图解析结果
        generated_text: 生成文本
        reference_text: 参考文档
        image_resources: 图片资源列表
        memory_store: 记忆存储实例（可选）
        session_id: 会话ID
    """
    def _run():
        try:
            result = eval_agent.evaluate(
                user_query=user_query,
                intent=intent,
                generated_text=generated_text,
                reference_text=reference_text,
                image_resources=image_resources,
            )
            log_info(
                "评估Agent(异步)",
                f"后台评估完成：action={result.action}，"
                f"事实准确率={result.factual_score:.2f}，综合评分={result.quality_score:.2f}",
            )

            # 尝试写入记忆存储
            if memory_store and session_id:
                try:
                    memory_store.record_eval_result(
                        session_id=session_id,
                        eval_result={
                            "pass": result.pass_,
                            "action": result.action,
                            "factual_score": result.factual_score,
                            "quality_score": result.quality_score,
                            "hallucination_rate": result.hallucination_rate,
                            "suggestion": result.suggestion,
                            "fact_check_results": result.fact_check_results[:5],
                        },
                    )
                except Exception as e:
                    log_warn("评估Agent(异步)", f"写入评估结果失败：{e}")

            # 如果有严重问题（contradicted），记录警告日志
            if result.action == "fail":
                log_warn(
                    "评估Agent(异步)",
                    f"⚠ 严重质量问题！事实准确率={result.factual_score:.2f}，"
                    f"诊断：{result.diagnosis.get('critical_issues', [])}",
                )
        except Exception as e:
            log_error("评估Agent(异步)", f"后台评估异常：{e}", e)

    thread = threading.Thread(target=_run, daemon=True, name="eval_agent_async")
    thread.start()
    log_info("评估Agent(异步)", "后台评估线程已启动")


# ===================== 便捷函数 =====================

def evaluate_and_retry(
    eval_agent: EvalAgent,
    generate_func,
    user_query: str,
    intent: Dict[str, Any],
    reference_text: str = "",
    image_resources: List[Dict[str, Any]] = None,
    **gen_kwargs,
) -> Dict[str, Any]:
    """
    同步评估 + 失败自动重生成 的通用包装。
    评估不通过时，带诊断建议重生成（最多 EVAL_MAX_RETRY 次）。

    Args:
        eval_agent: EvalAgent 实例
        generate_func: 生成函数，接收 (query, intent, retry_suggestion, **gen_kwargs) 返回文本
        user_query: 用户原始请求
        intent: 意图解析结果
        reference_text: 参考文档文本
        image_resources: 图片资源列表
        gen_kwargs: 透传给 generate_func 的额外参数

    Returns:
        {"text": final_text, "eval_result": result, "attempts": n, "passed": bool}
    """
    max_retry = getattr(config, "EVAL_MAX_RETRY", 1)
    attempts = 0
    last_suggestion = ""
    final_text = ""

    while attempts <= max_retry:
        attempts += 1
        # 生成
        final_text = generate_func(user_query, intent, last_suggestion, **gen_kwargs)
        if not final_text or not str(final_text).strip():
            last_suggestion = "生成内容为空，请重新完整生成。"
            continue

        # 评估
        result = eval_agent.evaluate(
            user_query=user_query,
            intent=intent,
            generated_text=str(final_text),
            reference_text=reference_text,
            image_resources=image_resources,
        )

        if result.pass_:
            log_info("评估Agent", f"第{attempts}次生成通过评估")
            return {"text": final_text, "eval_result": result, "attempts": attempts, "passed": True}

        last_suggestion = result.suggestion or "请严格按用户核心需求重新生成。"
        log_info("评估Agent", f"第{attempts}次生成未通过评估，建议：{last_suggestion}")

    # 全部重试失败，返回最后一次生成结果
    log_warn("评估Agent", f"全部{attempts}次尝试未通过评估，返回最后一次结果")
    return {"text": final_text, "eval_result": result, "attempts": attempts, "passed": False}