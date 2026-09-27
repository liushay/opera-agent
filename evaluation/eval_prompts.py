# evaluation/eval_prompts.py 评估Agent提示词模板
# 功能：
#   为评估Agent的5个步骤提供独立的提示词模板：
#     ① 事实声明提取器
#     ② 事实核查器（逐条验证）
#     ③ 意图对齐器
#     ④ 质量评分器（7维度）
#     ⑤ 决策路由（规则判断，无需LLM）
#   新增：
#     ⑥ Span级幻觉检测器（一次调用输出所有幻觉span）
#     ⑦ 批量事实核查器（一次调用核查所有声明）
#     ⑧ 修正指令生成器（可执行的修正操作列表）
# 设计原则：
#   - 所有提示词均使用独立评估LLM（deepseek-r1:7b），与生成Agent（qwen2.5:7b-instruct）完全解耦
#   - 每个模板独立职责，可单独调试和优化
#   - 输出格式严格JSON，便于下游解析
#   - 新版提示词（⑥⑦⑧）支持批量处理和Span级精确标注

import json
from typing import Any, Dict, List


# ===================== ① 事实声明提取器 =====================

def build_claim_extraction_prompt(generated_text: str) -> str:
    """
    ① 事实声明提取器提示词
    从生成结果中提取所有事实性声明（人名、术语、关系、属性描述等），
    每条声明是一个独立的可验证断言。
    """
    return f"""你是【事实声明提取器】。请从以下AI生成内容中提取所有事实性声明。

每条声明必须是一个独立的、可验证的断言。包括：
- 人名/角色名及其属性（如"关羽的脸谱是红色的"）
- 戏曲术语定义（如"西皮流水是一种京剧唱腔板式"）
- 历史事实描述（如"京剧形成于清代乾隆年间"）
- 因果关系断言（如"脸谱红色代表忠义"）
- 数量/时间/地点等具体信息

排除：
- 纯主观评价（如"非常精彩"）
- 闲聊/寒暄
- 问题反问（如"你想了解什么呢？"）
- 过渡性语句

【AI生成内容】
{generated_text[:5000]}

【输出格式】
严格输出JSON数组，每个元素是一条声明：
{{
  "claims": [
    "声明原文1",
    "声明原文2",
    ...
  ]
}}
只输出JSON，不输出任何其他文字。"""


# ===================== ② 事实核查器（逐条，旧版兼容） =====================

def build_fact_verification_prompt(claim: str, evidence: str) -> str:
    """
    ② 事实核查器提示词（旧版逐条核查，保留兼容）
    判断单条声明是否能被检索到的知识库资料支撑。
    """
    return f"""你是【事实核查员】。请判断以下声明是否能被参考资料支撑。

【待验证声明】
{claim}

【知识库检索到的参考资料】
{evidence[:3000] if evidence else "（无相关参考资料）"}

【判断标准】
- "verified"：参考资料明确支持该声明的全部内容
- "partial"：参考资料部分支持，但存在不完整或模糊之处
- "unverified"：参考资料中找不到对该声明的任何支撑
- "contradicted"：参考资料内容与该声明直接矛盾

【输出格式】
严格输出JSON：
{{
  "verdict": "verified|partial|unverified|contradicted",
  "reason": "判断依据（引用参考资料中的具体内容）",
  "confidence": 0.0到1.0之间的置信度
}}
只输出JSON，不输出任何其他文字。"""


# ===================== ③ 意图对齐器 =====================

def build_intent_alignment_prompt(
    user_query: str,
    core_task: str,
    must_require: List[str],
    forbid_list: List[str],
    generated_text: str,
) -> str:
    """
    ③ 意图对齐器提示词
    检查生成内容是否完全对齐用户意图（core_task + must + forbid）。
    """
    must_str = "\n".join(f"  - {m}" for m in must_require) if must_require else "  （无）"
    forbid_str = "\n".join(f"  - {f}" for f in forbid_list) if forbid_list else "  （无）"

    return f"""你是【意图对齐检查员】。请检查AI生成内容是否完全对齐用户意图。

【用户原始请求】
{user_query}

【本次核心任务】
{core_task}

【必须遵守的硬性要求】
{must_str}

【禁止行为】
{forbid_str}

【AI生成内容】
{generated_text[:4000]}

【检查维度】
1. 核心任务对齐：生成内容是否完全围绕核心任务，没有跑题或做多余的事
2. 硬性要求满足：是否满足所有 must_require
3. 禁止行为遵守：是否违反任何 forbid_list
4. 用户焦点优先：是否以用户当前请求为最高优先级

【输出格式】
严格输出JSON：
{{
  "aligned": true或false,
  "core_task_aligned": true或false,
  "must_satisfied": true或false,
  "forbid_respected": true或false,
  "user_focus_priority": true或false,
  "issues": ["具体问题描述1", "问题描述2"],
  "suggestion": "重生成时应修正的要点"
}}
只输出JSON，不输出任何其他文字。"""


# ===================== ④ 质量评分器（7维度） =====================

def build_quality_scoring_prompt(
    user_query: str,
    generated_text: str,
    fact_check_results: List[Dict[str, Any]],
    intent_alignment: Dict[str, Any],
    image_resources: List[Dict[str, Any]] = None,
) -> str:
    """
    ④ 质量评分器提示词（7维度）
    综合所有评估信息，给出7维度评分和综合诊断。
    """
    # 汇总事实核查统计
    total_claims = len(fact_check_results)
    verified_count = sum(1 for r in fact_check_results if r.get("verdict", {}).get("verdict") == "verified")
    partial_count = sum(1 for r in fact_check_results if r.get("verdict", {}).get("verdict") == "partial")
    unverified_count = sum(1 for r in fact_check_results if r.get("verdict", {}).get("verdict") == "unverified")
    contradicted_count = sum(1 for r in fact_check_results if r.get("verdict", {}).get("verdict") == "contradicted")

    fact_summary = (
        f"总声明数：{total_claims}，已验证：{verified_count}，"
        f"部分验证：{partial_count}，未验证：{unverified_count}，矛盾：{contradicted_count}"
    )

    # 图像资源检查
    image_info = ""
    if image_resources:
        has_image = any(
            r.get("type") == "image" and r.get("image_status") == "generated"
            for r in image_resources
        )
        image_info = f"\n图像资源：{'已生成真实图片' if has_image else '无真实图片资源'}"

    return f"""你是【质量评分员】。请对AI生成内容进行7维度综合评分（每个维度1-5分）。

【用户原始请求】
{user_query}

【AI生成内容】
{generated_text[:4000]}

【事实核查统计】
{fact_summary}
{image_info}

【意图对齐结果】
{json.dumps(intent_alignment, ensure_ascii=False)[:500]}

【7个评分维度】
1. factual_accuracy（事实准确性）：生成内容中的事实声明是否与知识库一致。有contradicted扣分严重，unverified扣分中等
2. intent_alignment（意图对齐度）：是否完全响应用户核心任务，没有跑题
3. completeness（完整性）：是否覆盖用户问题的所有子问题
4. hallucination_free（无幻觉率）：是否存在知识库没有的编造内容。unverified和contradicted比例越高，此项越低
5. reference_accuracy（引用准确性）：引用来源是否正确、可追溯
6. task_completion（任务完成度）：多模态任务是否产生真实产出（如图片是否生成）
7. user_satisfaction（用户满意度预估）：回答是否直接、清晰、有用

【输出格式】
严格输出JSON：
{{
  "scores": {{
    "factual_accuracy": 5,
    "intent_alignment": 5,
    "completeness": 4,
    "hallucination_free": 5,
    "reference_accuracy": 5,
    "task_completion": 5,
    "user_satisfaction": 4
  }},
  "overall": 4.5,
  "critical_issues": ["严重问题1", "严重问题2"],
  "minor_issues": ["小问题1"],
  "suggestion": "重生成时的修正建议"
}}
只输出JSON，不输出任何其他文字。"""


# ===================== ⑥ 新增: Span级幻觉检测提示词 =====================

def build_span_hallucination_prompt(
    generated_text: str,
    factual_context: str,
    user_query: str = "",
) -> str:
    """
    Span级幻觉检测器提示词（核心创新）。
    一次性输出所有幻觉的精确位置和修正内容，而非逐条声明分别调用LLM。
    输出格式为 span 级标注，每个幻觉标记包含：起始位置、结束位置、修正文本、证据引用。

    与旧版 build_fact_verification_prompt 的区别：
    - 旧版：逐条声明 -> 逐条调用LLM（N次调用，耗时N*20秒）
    - 新版：一次LLM调用同时输出所有幻觉span（1次调用，约30秒）
    """
    prompt = f"""你是【Span级幻觉检测员】。请逐句检查AI生成内容，标记所有与知识库事实不符的幻觉段落。

【用户原始请求】
{user_query[:500] if user_query else "（无）"}

【知识库参考资料（事实依据）】
{factual_context[:5000] if factual_context else "（无可用参考资料）"}

【AI生成内容】
{generated_text[:5000]}

【任务】
逐句比对AI生成内容与知识库参考资料，找出所有：
1. contradicted（矛盾）：与知识库明确矛盾的内容
2. unverified（无法验证）：知识库中没有对应支撑的内容
3. partial（部分正确）：部分正确但存在偏差的内容

对每个问题段落，输出精确的文本片段及其修正值。

【输出格式】
严格输出JSON：
{{
  "hallucinations": [
    {{
      "span": "原文中有问题的文本片段",
      "verdict": "contradicted|unverified|partial",
      "correct": "修正后的正确内容（contradicted时必填）",
      "reason": "判定依据（引用知识库中的具体内容）",
      "confidence": 0.0到1.0之间的置信度
    }}
  ],
  "overall_assessment": {{
    "total_claims": 0,
    "contradicted_count": 0,
    "unverified_count": 0,
    "partial_count": 0,
    "verified_count": 0,
    "hallucination_severity": "none|low|medium|high|critical",
    "summary": "一句话总结幻觉情况"
  }}
}}
只输出JSON，不输出任何其他文字。"""
    return prompt


# ===================== ⑦ 新增: 批量事实核查提示词 =====================

def build_batch_fact_check_prompt(
    claims: List[str],
    evidence_map: Dict[str, str],
) -> str:
    """
    批量事实核查提示词（替代旧版逐条核查）。
    将多条声明和对应证据一次性提交给LLM，一次调用完成所有核查。

    与旧版 build_fact_verification_prompt 的区别：
    - 旧版：每条声明单独调用一次LLM（N条声明=N次调用）
    - 新版：所有声明+证据一次性提交（1次调用）
    """
    claims_text = "\n".join(
        f"[{i+1}] {claim}" for i, claim in enumerate(claims)
    )
    evidence_text = "\n\n".join(
        f"--- 声明[{i+1}]的参考资料 ---\n{evidence_map.get(str(i), '（无）')[:500]}"
        for i in range(len(claims))
    ) if evidence_map else "（无参考资料）"

    prompt = f"""你是【批量事实核查员】。请一次性核查以下所有声明是否被参考资料支撑。

【待核查声明】
{claims_text[:4000]}

【参考资料】
{evidence_text[:4000]}

【判断标准】
- "verified"：参考资料明确支持该声明的全部内容
- "partial"：参考资料部分支持，但存在不完整或模糊之处
- "unverified"：参考资料中找不到对该声明的任何支撑
- "contradicted"：参考资料内容与该声明直接矛盾

【输出格式】
严格输出JSON：
{{
  "results": [
    {{
      "claim_index": 0,
      "claim": "声明原文",
      "verdict": "verified|partial|unverified|contradicted",
      "reason": "判断依据",
      "confidence": 0.0,
      "correction": "修正后的正确内容（contradicted时必填）"
    }}
  ],
  "summary": {{
    "total": 0,
    "verified": 0,
    "partial": 0,
    "unverified": 0,
    "contradicted": 0
  }}
}}
只输出JSON，不输出任何其他文字。"""
    return prompt


# ===================== ⑧ 新增: 修正指令生成提示词 =====================

def build_correction_instruction_prompt(
    user_query: str,
    generated_text: str,
    span_issues: List[Dict[str, Any]],
    intent_alignment: Dict[str, Any],
    quality_scores: Dict[str, Any],
) -> str:
    """
    修正指令生成器提示词。
    将评估诊断结果转化为生成Agent能直接执行的结构化修正指令。
    输出格式为精确的修正操作列表（replace/delete/append/rewrite）。
    """
    issues_text = json.dumps(span_issues, ensure_ascii=False, indent=2)[:3000]
    intent_text = json.dumps(intent_alignment, ensure_ascii=False)[:500]
    scores_text = json.dumps(quality_scores, ensure_ascii=False)[:500]

    prompt = f"""你是【修正指令生成器】。请根据评估诊断结果，为生成Agent生成精确、可执行的修正指令。

【用户原始请求】
{user_query[:500]}

【当前生成内容】
{generated_text[:3000]}

【幻觉检测结果（Span级）】
{issues_text}

【意图对齐结果】
{intent_text}

【质量评分】
{scores_text}

【任务】
生成一组精确的修正操作指令。每个指令必须包含：
- action: replace（替换文本）/ delete（删除文本）/ append（追加内容）/ rewrite（重写整段）
- 对replace：指定 old_text（被替换的原文）和 new_text（替换后的内容）
- 对delete：指定 target_text（要删除的原文）
- 对append：指定 content（要追加的内容）和 position（追加位置：beginning/end）
- 对rewrite：指定 section（要重写的段落描述）和 direction（重写方向）
- 每个指令附带 reason（修正原因）

【输出格式】
严格输出JSON：
{{
  "corrections": [
    {{
      "action": "replace|delete|append|rewrite",
      "old_text": "要替换的原文片段",
      "new_text": "替换后的内容",
      "target_text": "要删除的原文",
      "content": "要追加的内容",
      "position": "beginning|end",
      "section": "要重写的段落描述",
      "direction": "重写方向",
      "reason": "修正原因"
    }}
  ],
  "regenerate_hint": "给生成Agent的整体重生成提示（一句话）",
  "priority": "high|medium|low"
}}
只输出JSON，不输出任何其他文字。"""
    return prompt