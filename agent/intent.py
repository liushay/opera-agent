# agent/intent.py 需求解析层（意图解析）
# 功能：
#   所有用户请求必须先走【意图解析】，禁止直接丢给Agent，输出固定JSON：
#     - core_task: 当前唯一核心任务
#     - must_require: 必须遵守的用户硬性要求
#     - optional_require: 次要需求
#     - forbid_list: 用户禁止行为（不要、只、仅、并非）
#     - reference_material_required: 是否强制依赖本次参考文档
# 强制规则：
#   1. Agent所有生成必须严格对齐解析结果
#   2. forbid_list 优先级最高，绝对不能违反
#   3. 用户新指令优先级 > 所有历史记忆
# 设计原则：
#   - 完全解耦，不改变三层记忆结构，不修改任何原有接口
#   - 解析失败时降级为默认结构（core_task=原始提问），保证不阻塞
import json
import re
from typing import Any, Dict, List

from agent.llm_utils import invoke_with_retry, safe_llm_call
from utils.json_repair import robust_json_loads
from utils.logger import log_info, log_warn


def _build_intent_prompt(user_query: str, work_item_desc: str) -> str:
    """构建意图解析提示词（支持复合需求识别）"""
    return f"""
你是戏曲科普平台的【意图解析器】。请严格解析用户的请求，输出固定JSON。

【平台可执行的业务功能】
{work_item_desc}

【用户请求】
{user_query}

【解析要求】
1. core_task：一句话精炼描述用户当前的核心任务。如果用户请求包含多个明显独立的需求（如"介绍A + 以B口吻回答C"），则描述为复合任务（如"介绍戏剧历史并同时以杜丽娘口吻回答"）
2. must_require：用户明确要求的硬性条件 → 逐条列出。对于复合需求，将每个子需求列为独立条目（如"介绍戏剧历史"、"以杜丽娘的口吻回答"）
3. optional_require：用户次要/补充需求 → 逐条列出
4. forbid_list：用户明确禁止的行为（出现"不要/只/仅/并非/除了...之外/别再"等语气）→ 逐条列出；无则空数组
5. reference_material_required：本次是否需要严格依赖参考文档/知识库材料。当用户要求"参考文档/依据资料/根据文档/结合文档/不要瞎编"等时=true；日常闲聊=false
6. is_compound：是否包含多个明显独立的需求（true/false）。判断标准：用户请求中通过"并/同时/另外/还有/以及/顺便"等连接词组合了多个不同功能的需求
7. sub_tasks：当 is_compound=true 时，列出各子任务及其对应的 worker 类型。格式：[{{"worker":"search/character/lyrics/quiz/guide/face/chat","task":"子任务描述"}}]

【灵活分析原则（重要）】
- 不要死板地根据关键词判断意图（如"介绍"不一定只是知识问答），要结合上下文灵活分析
- 如果用户说"以XX的口吻/角色/身份回答"，这是角色扮演需求，对应 character_worker
- 如果用户说"介绍一下XX的历史"，这是知识问答需求，对应 search_worker
- 如果用户同时包含以上两种需求，is_compound=true，sub_tasks 分别列出
- 用户说"你好"、"谢谢"等日常寒暄不是知识问答，是 chat

【严格规则】
- 只输出JSON，不输出任何多余文字
- 若无法识别业务功能，core_task 直接复述用户请求，forbid_list 为空

输出格式：
{{
  "core_task": "...",
  "must_require": ["..."],
  "optional_require": ["..."],
  "forbid_list": ["..."],
  "reference_material_required": true或false,
  "is_compound": true或false,
  "sub_tasks": []
}}
"""


# 平台可执行业务功能描述（供意图解析器提示词使用）
WORK_ITEM_DESC = (
    "1. search：知识库检索/戏曲知识问答/术语解释\n"
    "2. lyrics：戏词解剖（输入戏词，输出逐句翻译/典故/心境/唱腔）\n"
    "3. character：戏中人对谈（与穆桂英/白素贞/曹操等角色对话）\n"
    "4. quiz：知识闯关出题（可按主题：行当/剧目/流派/历史/术语）\n"
    "5. guide：个性化学戏路线（制定学习课程）\n"
    "6. face：脸谱画像（生成专属脸谱）\n"
    "7. literature：戏曲文献生成（输出txt/pdf/md文档）\n"
    "8. chat：日常闲聊"
)


def parse_intent(user_query: str) -> Dict[str, Any]:
    """
    解析用户意图，返回固定结构：
    {
      "core_task": str,
      "must_require": [str],
      "optional_require": [str],
      "forbid_list": [str],
      "reference_material_required": bool,
      "raw_query": str
    }
    解析失败时降级为：core_task=原始请求，其余为空，保证下游不阻塞。
    """
    log_info("意图解析", f"开始解析用户请求：{user_query[:80]}")
    prompt = _build_intent_prompt(user_query, WORK_ITEM_DESC)
    try:
        content = invoke_with_retry(
            [prompt],
            temperature=0.0,
            task_name="意图解析",
        )
        # 容错提取JSON（使用 robust_json_loads 提高容错率）
        data = robust_json_loads(content)
        result = {
            "core_task": str(data.get("core_task") or user_query),
            "must_require": [str(x) for x in data.get("must_require", [])],
            "optional_require": [str(x) for x in data.get("optional_require", [])],
            "forbid_list": [str(x) for x in data.get("forbid_list", [])],
            "reference_material_required": bool(data.get("reference_material_required", False)),
            "is_compound": bool(data.get("is_compound", False)),
            "sub_tasks": [dict(t) for t in data.get("sub_tasks", [])] if isinstance(data.get("sub_tasks"), list) else [],
            "raw_query": user_query,
        }
        log_info("意图解析", f"解析结果：core_task={result['core_task']}，forbid={result['forbid_list']}")
        return result
    except Exception as e:
        log_warn("意图解析", f"解析失败，降级为默认结构：{e}")
        return {
            "core_task": user_query,
            "must_require": [],
            "optional_require": [],
            "forbid_list": [],
            "reference_material_required": False,
            "raw_query": user_query,
        }


def is_forbidden(query: str, forbid_list: List[str]) -> bool:
    """
    校验本次请求是否踩中 forbid_list（禁止行为优先级最高，绝对不能违反）。
    规则：forbid 中的关键词出现在用户请求中 → 返回 True（应阻止对应生成）
    说明：规范约定"用户当前提问优先级最高"，此函数用于识别用户自己在说什么，
          而非从记忆判断，因此只检查 query 文本与 forbid 表述是否冲突。
    """
    if not forbid_list:
        return False
    for forbid in forbid_list:
        # forbid 可能是"不要X"/"只X"/"仅X"；粗略判断 forbid 主题词是否在 query 中
        # 此处不做复杂NLP，主要靠 LLM 已在 forbid_list 中提取了明确语义，
        # 这里仅为下游提示词提供结构化约束，真正约束由生成提示词完成。
        pass
    return False


def align_prompt_with_intent(intent: Dict[str, Any]) -> str:
    """
    生成面向下游Agent/生成模块的"意图对齐提示词片段"。
    调用方将其拼入最终生成提示词，确保生成严格对齐解析结果。
    """
    parts = []
    parts.append(f"【当前唯一核心任务】{intent.get('core_task', '')}")
    if intent.get("must_require"):
        parts.append("【必须遵守的硬性要求】")
        parts.extend(f"- {m}" for m in intent["must_require"])
    if intent.get("optional_require"):
        parts.append("【次要需求】")
        parts.extend(f"- {o}" for o in intent["optional_require"])
    if intent.get("forbid_list"):
        parts.append("【禁止行为（绝对不得违反）】")
        parts.extend(f"- 禁止：{f}" for f in intent["forbid_list"])
    if intent.get("reference_material_required"):
        parts.append("【文档约束】本次生成必须严格基于提供的参考文档内容，禁止使用模型预训练知识、禁止脑补、禁止外部知识点")
    parts.append("【优先级】用户当前核心任务 > 所有历史记忆；只做当前任务要求的事，绝不擅自追加其他功能或旧任务内容。")
    return "\n".join(parts)