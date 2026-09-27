# agent/output_validator.py 输出校验层
# 功能：
#   每次生成结束后自动校验：
#     1. 是否完全符合用户当前核心需求
#     2. 是否遵守所有 must / forbid
#     3. 是否没有混入旧任务记忆内容
#     4. 是否严格基于参考文档
#   不满足 → 返回校验结果供上层自动重生成（最多重试 MAX_VALIDATE_RETRY 次）
# 设计原则：
#   - 完全解耦，不改变三层记忆结构，不修改任何原有接口
#   - 校验失败时返回结构化结果，由调用方决定是否重生成
import json
import re
from typing import Any, Dict, List

from agent.llm_utils import invoke_with_retry, safe_llm_call
from agent.intent_guard import is_image_intent
from utils.logger import log_info, log_warn

# 输出校验最大重试次数（规范约定不无限循环）
MAX_VALIDATE_RETRY = 2


def build_validate_prompt(
    user_query: str,
    core_task: str,
    must_require: List[str],
    forbid_list: List[str],
    output_text: str,
    reference_required: bool,
    reference_text: str = "",
) -> str:
    """构建输出校验提示词"""
    ref_part = ""
    if reference_required:
        ref_part = f"\n【本次参考文档】\n{reference_text[:3000] if reference_text else '（无参考文档，仅能基于知识库检索结果）'}"
    return f"""你是【输出质量校验员】。请校验以下AI生成内容是否完全符合用户需求，输出严格JSON。

【用户原始需求】{user_query}
【本次唯一核心任务】{core_task}
【必须遵守】
{chr(10).join('- ' + m for m in must_require) if must_require else '（无）'}
【禁止行为】
{chr(10).join('- ' + f for f in forbid_list) if forbid_list else '（无）'}
{ref_part}

【AI生成内容】
{output_text[:4000]}

【校验维度】
1. task_aligned：内容是否完全对齐核心任务，没有跑题/多余功能（尤其不得混入旧任务的闯关、文献等内容）
2. must_satisfied：是否满足所有必须要求
3. forbid_respected：是否违反任何禁止行为
4. reference_based（当 reference_required=true 时）：内容是否严格基于参考文档，无模型编造/私货
5. user_current_focus：是否以用户当前提问为最高优先级，没有忽略新需求

【输出格式】
{{
  "pass": true或false,
  "task_aligned": true或false,
  "must_satisfied": true或false,
  "forbid_respected": true或false,
  "reference_based": true或false,
  "user_current_focus": true或false,
  "issues": ["未通过维度的具体问题"],
  "suggestion": "重生成时应修正的要点"
}}
只输出JSON。"""


def validate_output(
    user_query: str,
    intent: Dict[str, Any],
    output_text: str,
    reference_text: str = "",
    image_resources: List[Dict[str, Any]] = None,
    face_worker_dispatched: bool = False,
) -> Dict[str, Any]:
    """
    校验生成内容是否合格。
    Args:
        user_query: 用户原始请求
        intent: 意图解析结果
        output_text: 生成文本内容
        reference_text: 参考文档（可选）
        image_resources: 本次返回报文中的图片资源列表（可选，新增）
             当用户有图像生成诉求但结果没有真实图片资源时，校验必须失败，
             由上层重新派发图像生成 worker（需求2）。
        face_worker_dispatched: 是否已调度 face_worker 执行脸谱子任务（新增）
             当复合请求中包含 face_worker 时，硬规则强制 image_required=True，
             不依赖 LLM 的 is_image_intent 判断（LLM 可能因知识问答信号词误判）。
    Returns:
      {"pass": bool, "issues": [...], "suggestion": str, "checks": {...},
       "image_required": bool, "image_missing": bool}
    """
    core_task = intent.get("core_task", user_query)
    must = intent.get("must_require", [])
    forbid = intent.get("forbid_list", [])
    ref_required = bool(intent.get("reference_material_required", False))

    # ===== 修复：图像诉求硬性校验 =====
    # 用户明确要求生成图片/画图/生成脸谱，但结果没有图片资源 → 校验必须失败
    # （不能假装任务完成，需触发上层重新调度图像生成 worker）
    #
    # 硬规则（修复）：只要调度列表中包含 face_worker，强制 image_required=True，
    # 忽略 LLM 的 is_image_intent 判断（复合请求如"介绍曹操并生成脸谱"中，
    # is_image_intent 可能因"介绍"知识信号词误判为 False）。
    llm_image_required = is_image_intent(user_query)
    if face_worker_dispatched:
        image_required = True
        log_info(
            "输出校验",
            f"硬规则覆盖：face_worker 已调度，强制 image_required=True"
            f"（LLM 判断 image_required={llm_image_required}，已被硬规则覆盖）",
        )
    else:
        image_required = llm_image_required
        log_info(
            "输出校验",
            f"image_required 使用 LLM 判断：{image_required}"
            f"（face_worker_dispatched=False，未触发硬规则覆盖）",
        )
    image_resources = image_resources or []
    has_real_image = any(
        r.get("type") == "image"
        and r.get("image_status") == "generated"
        and r.get("image_url")
        for r in image_resources
    )
    image_missing = bool(image_required and not has_real_image)

    prompt = build_validate_prompt(
        user_query, core_task, must, forbid, output_text, ref_required, reference_text,
    )
    try:
        content = invoke_with_retry([prompt], temperature=0.0, task_name="输出校验")
        start = content.find("{")
        end = content.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("无有效JSON")
        data = json.loads(content[start:end])
        passed = bool(data.get("pass", False))
        # 图像缺失时强制 pass=False（LLM 文本校验通过也不能算完成）
        if image_missing:
            passed = False
        issues = [str(i) for i in data.get("issues", [])]
        if image_missing:
            issues.insert(0, "用户需要图片，但结果没有图片资源（image_resources为空或未生成真实图片）")
        result = {
            "pass": passed,
            "issues": issues,
            "suggestion": str(data.get("suggestion", "")),
            "checks": {
                "task_aligned": bool(data.get("task_aligned", False)),
                "must_satisfied": bool(data.get("must_satisfied", False)),
                "forbid_respected": bool(data.get("forbid_respected", False)),
                "reference_based": bool(data.get("reference_based", False)),
                "user_current_focus": bool(data.get("user_current_focus", False)),
            },
            # 新增：图片诉求标记（供上层决定重试调度策略）
            "image_required": image_required,
            "image_missing": image_missing,
        }
        log_info(
            "输出校验",
            f"校验结果 pass={result['pass']}，issues={result['issues']}"
            f"，image_required={image_required}，image_missing={image_missing}",
        )
        return result
    except Exception as e:
        # 修复(Bug)：即使 LLM 校验调用失败，image_missing 也是纯规则判断（不依赖LLM）。
        # 用户有图像诉求但结果无图片资源 → 必须返回 pass=False，交给上层重试调度，
        # 不能因 LLM 校验失败而误放行（否则会假装任务完成）。
        issues = []
        if image_missing:
            issues.append("用户需要图片，但结果没有图片资源（image_resources为空或未生成真实图片）")
        log_warn(
            "输出校验",
            f"校验调用失败，image_missing={image_missing}，pass={not image_missing}（避免阻塞主流程）：{e}",
        )
        return {
            "pass": not image_missing,
            "issues": issues,
            "suggestion": "用户需要图片，请重新派发图像生成worker并产出真实图片资源。" if image_missing else "",
            "checks": {},
            "image_required": image_required,
            "image_missing": image_missing,
        }


def regenerate_on_failure(
    generate_func,
    user_query: str,
    intent: Dict[str, Any],
    reference_text: str = "",
    max_retry: int = MAX_VALIDATE_RETRY,
    **gen_kwargs,
) -> Any:
    """
    生成 + 校验 + 失败自动重生成 的通用包装。
    Args:
        generate_func: 生成函数，接收 (query, intent, retry_feedback, **gen_kwargs) 返回文本
        user_query: 用户原始请求
        intent: 意图解析结果
        reference_text: 参考文档文本
        max_retry: 最大重生成次数（默认2）
        gen_kwargs: 透传给 generate_func 的额外参数
    Returns:
        (final_text, attempts, passed)
    """
    attempts = 0
    last_suggestion = ""
    while attempts <= max_retry:
        attempts += 1
        output_text = generate_func(user_query, intent, last_suggestion, **gen_kwargs)
        # 生成失败（空/异常）直接重试
        if not output_text or not str(output_text).strip():
            last_suggestion = "生成内容为空，请重新完整生成。"
            continue
        result = validate_output(user_query, intent, str(output_text), reference_text)
        if result["pass"]:
            return output_text, attempts, True
        last_suggestion = result.get("suggestion", "请严格按用户核心需求重新生成。")
        log_info("输出校验", f"第{attempts}次生成未通过，将重生成。建议：{last_suggestion}")
    return output_text, attempts, False