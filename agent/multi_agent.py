# agent/multi_agent.py 多智能体协作（主管-工人模式）
# 功能：意图识别 → 分派角色化Worker → 黑板共享成果 → 汇总回答
#
# Worker 清单（角色化，取代原 search/calc 双工人）：
#   - search_worker     知识检索工人（原保留）
#   - lyrics_worker     戏词解剖工人（戏词解剖室）
#   - character_worker  人物对谈工人（戏中人对谈）
#   - quiz_worker       知识闯关出题工人
#   - guide_worker      学戏路线工人
#   - face_worker       脸谱画像工人

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from typing import TypedDict, Annotated, Sequence, Literal
import operator
import json
import httpx

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

import config
from utils.logger import log_info, log_warn, log_error
from utils.exception_handler import global_exception_handler

# ---- 戏曲科普业务模块 ----
from opera import annotate_lyrics, character_chat, generate_quiz, generate_course, generate_face_profile
from rag.vectorstore import hybrid_retrieve
from utils.rag_exceptions import LLMModelException, AgentFlowException, VectorStoreException, BM25IndexException
from agent.memory import memory_manager
from agent.intent import align_prompt_with_intent
from agent.task_context import get_current_task_only


def get_multi_llm():
    try:
        return ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMP)
    except httpx.ConnectError as e:
        err_msg = "多智能体LLM连接Ollama失败"
        log_error("多Agent LLM初始化失败", err_msg, e)
        raise LLMModelException(err_msg, e)
    except Exception as e:
        err_msg = "多智能体LLM实例创建失败"
        log_error("多Agent LLM异常", err_msg, e)
        raise LLMModelException(err_msg, e)


llm = get_multi_llm()
MAX_RETRY = 2


class MultiAgentState(TypedDict):
    user_query: str
    sub_task_list: list
    worker_result: list
    messages: Annotated[Sequence[BaseMessage], operator.add]
    retry_times: int
    need_retry: bool
    # 记忆增强字段（可选，不影响原有调用方）
    mem_session_id: str
    mem_task_id: str
    # 意图解析结果（新增，可选字段兼容旧调用方）
    intent: dict
    reference_text: str
    # 黑板：各Worker产出在此汇聚，供后续Worker/汇总读取
    blackboard: Annotated[dict, operator.add]
    # 图片资源列表（新增，可选字段兼容旧调用方）：
    # 图像类 Worker 执行工具后产出真实图片路径，汇总节点合并进最终返回报文供前端渲染。
    image_resources: list
    # ===== 反思增强字段（反馈驱动重规划） =====
    last_search_query: str          # 上次 search_worker 检索的 query
    last_search_feedback: dict      # 上次检索反思的结构化诊断报告
    accumulated_worker_result: str  # 多轮 search_worker 结果累积
    # ===== 复合需求多 worker 串行调度 =====
    _task_index: int                # 当前正在执行的 worker 在 sub_task_list 中的索引
    _compound_workers: list         # 复合需求中待执行的 worker 列表


DEFAULT_BLACKBOARD = {}


# ===================== 节点1 主管（意图识别与任务分派） =====================
# 问题3修复：
#   1. 接入已解析的用户意图（intent），用户当前核心任务 = 最高优先级
#   2. 强制只输出一个最匹配的 worker，禁止自动追加用户未提出的任务（如没提闯关绝不生成闯关）
#   3. 增加规则化意图校验：根据 user_query 关键词过滤不允许的 worker
# 说明：意图校验纯函数已提取到 agent/intent_guard.py（轻量、可单测、不触发重型导入）
from agent.intent_guard import (
    filter_tasks_by_intent as _filter_tasks_by_intent,
    is_knowledge_query as _is_knowledge_query,
    validate_worker_against_query as _validate_worker_against_query,
    normalize_face_image_result as _normalize_face_image_result,
    build_face_reply_text as _build_face_reply_text,
)


@global_exception_handler
def supervisor_node(state: MultiAgentState) -> MultiAgentState:
    query = state["user_query"]
    log_info("多Agent主管节点", f"开始意图识别与任务分派，用户请求：{query}")
    mem_session_id = state.get("mem_session_id", config.DEFAULT_SESSION_ID)
    intent = state.get("intent") or {}
    # 意图约束片段：用户当前核心任务 = 最高优先级
    intent_constraint = ""
    if intent.get("core_task"):
        intent_constraint = f"\n【用户本次核心任务（最高优先级）】{intent.get('core_task', '')}\n只派遣与该核心任务最匹配的一个 worker，绝不额外追加其他功能。"
    if intent.get("forbid_list"):
        intent_constraint += "\n【用户禁止行为】" + "；".join(f"- {f}" for f in intent["forbid_list"])

    try:
        memory_manager.record_thinking(mem_session_id, f"多Agent主管识别意图：{query}")
    except Exception:
        pass

    # ===== 反思增强：注入上次检索反思反馈，驱动差异化重规划 =====
    retry_times = state.get("retry_times", 0)
    last_search_query = state.get("last_search_query", "")
    reflection_feedback = ""
    if retry_times > 0 and last_search_query:
        reflection_feedback = f"""
【上一轮检索反思——请据此调整检索策略】
- 上次检索query（禁止重复使用）："{last_search_query}"
- 当前重试次数：第{retry_times}次

【重试硬约束——检索主题必须锚定用户原始问题】
用户原始问题："{query}"
新检索query必须满足：
1. 必须包含用户原始问题的核心主题词
2. 只能在此基础上增加细节、角度、具体化，不能改变主题
3. 禁止将query替换为完全不同的主题
4. 禁止使用与上次完全相同的检索query
5. 可以尝试：拆分查询、换同义词、从具体人物/剧目角度切入
"""
    # ===== 反思增强结束 =====

    prompt = f"""
你是戏曲科普平台的任务调度主管。请判断用户请求属于哪类任务，并输出严格JSON。
{intent_constraint}
{reflection_feedback}

 【可派遣工人及适用场景】
1. search_worker：用户提问需要知识库检索来回答（戏曲知识问答、术语解释等）
2. lyrics_worker：用户输入了一段戏词/唱词，需要逐句翻译、典故考据、心境解读
3. character_worker：用户想和戏曲人物聊天/对谈（如"和穆桂英聊聊"）
4. quiz_worker：用户想答题闯关/出题测试
5. guide_worker：用户想学习某剧种，需要制定学习课程/路线
6. face_worker：用户想生成专属脸谱画像/脸谱解读（会调用即梦AI图像生成工具产出真实图片）
7. none：日常闲聊或无法确定任务类型

【强制规则】
- 只选择一个最匹配的 worker，绝对不要输出多个
- 用户没有明确提出的任务，绝不自动追加（例如用户只问脸谱，绝不额外生成闯关题目）
- 如果用户请求是知识问答/介绍类，使用 search_worker
- 只有用户明确说"出题/闯关/答题"时才使用 quiz_worker
- 【图像诉求强制 face_worker】只要用户请求包含"生成/画/绘制/设计 + 图片/画像/脸谱/图"类图像诉求
  （例如"生成一张越剧有关的脸谱"、"画一幅京剧脸谱"、"生成脸谱图片"），
  必须选择 face_worker 并调用图像工具，绝不能只输出文字描述代替真实图片。

【输出格式】
{{
  "worker": "choose_one_from_above",
  "task": "分配给工人的明确任务描述",
  "params": {{"补充参数，如人物ID、主题、难度等，没有则空对象}}
}}

用户请求：{query}
只输出JSON。
"""
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
    except httpx.ConnectError as e:
        err_msg = "主管节点调用Ollama连接失败"
        log_error("多Agent主管LLM异常", err_msg, e)
        raise LLMModelException(err_msg, e)
    except Exception as e:
        err_msg = "主管任务识别流程失败"
        log_error("多Agent主管流程异常", err_msg, e)
        raise AgentFlowException(err_msg, e)

    try:
        content = resp.content.strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        task_data = json.loads(content[start:end]) if start >= 0 and end > start else []
    except json.JSONDecodeError as e:
        log_warn("多Agent主管", "模型输出JSON解析失败，降级为检索任务", e)
        task_data = {"worker": "search_worker", "task": query, "params": {}}

    if not isinstance(task_data, list):
        task_data = [task_data]
    valid_workers = {"search_worker", "lyrics_worker", "character_worker", "quiz_worker", "guide_worker", "face_worker"}
    has_none = any(t.get("worker") == "none" for t in task_data)
    if has_none:
        task_data = []
    filtered = []
    for t in task_data:
        if t.get("worker") in valid_workers:
            filtered.append(t)
    # ===== 问题3修复：意图校验，禁止追加用户未提出的任务 =====
    # 复合需求时传入 intent，支持多 worker 分派
    filtered = _filter_tasks_by_intent(filtered, query, intent)
    log_info("多Agent主管节点", f"任务分派完成（意图校验后）：{filtered}")
    return {"sub_task_list": filtered, "need_retry": False}


# ===================== Worker 节点 =====================

def _get_session(state: MultiAgentState) -> str:
    return state.get("mem_session_id", config.DEFAULT_SESSION_ID)


@global_exception_handler
def search_worker(state: MultiAgentState) -> MultiAgentState:
    """知识检索工人：对分派的任务执行混合检索，支持结果累积与反思重试
    修复死循环：短结果也触发重试 + 递增计数器 + 重试时跳过缓存"""
    task_list = state["sub_task_list"]
    outputs = list(state["worker_result"])
    need_retry_flag = False
    retry_times = state.get("retry_times", 0)
    mem_session_id = _get_session(state)
    is_retry = retry_times > 0
    log_info("多Agent检索工人", f"开始执行检索类任务，当前重试次数={retry_times}")
    extra_fields = {}

    for task in task_list:
        if task["worker"] != "search_worker":
            continue
        search_query = task.get("task", state["user_query"])
        try:
            memory_manager.record_tool_call(mem_session_id, "search_worker", search_query)
            # 重试时跳过缓存，强制真实检索，避免反复命中缓存拿到相同短文档
            ans = hybrid_retrieve(search_query, skip_cache=is_retry)
        except (VectorStoreException, BM25IndexException) as e:
            err_msg = f"检索任务[{search_query}]底层检索异常"
            log_error("多Agent检索工人异常", err_msg, e)
            raise AgentFlowException(err_msg, e)

        text_out = "\n".join([doc.page_content for doc in ans])
        result_len = len(text_out.strip())

        # 判断是否需要重试：结果为空 或 结果过短（<50字）
        if result_len == 0:
            log_warn("多Agent检索工人", f"任务:{search_query}检索结果为空，当前重试次数={retry_times}")
            if retry_times < MAX_RETRY:
                need_retry_flag = True
            else:
                log_warn("多Agent检索工人", f"检索结果为空且已达最大重试上限{MAX_RETRY}，强制终止重试")
        elif result_len < 50:
            log_warn("多Agent检索工人",
                     f"任务:{search_query}检索结果过短({result_len}字)，当前重试次数={retry_times}")
            if retry_times < MAX_RETRY:
                need_retry_flag = True
            else:
                log_warn("多Agent检索工人",
                         f"检索结果过短且已达最大重试上限{MAX_RETRY}，强制终止重试，直接使用当前结果")
        else:
            log_info("多Agent检索工人", f"任务:{search_query}检索结果充足({result_len}字)，无需重试")

        # 无论是否重试，都累积当前结果（保证重试上限后仍有结果可用）
        if retry_times > 0:
            previous_accumulated = state.get("accumulated_worker_result", "")
            if previous_accumulated:
                accumulated = (
                    previous_accumulated
                    + f"\n\n--- 补充检索结果（第{retry_times}次重试，query: {search_query}）---\n"
                    + text_out
                )
            else:
                accumulated = text_out
        else:
            accumulated = text_out
        extra_fields["accumulated_worker_result"] = accumulated
        extra_fields["last_search_query"] = search_query

        outputs.append({"worker": "search_worker", "task": search_query, "result": accumulated})
        memory_manager.record_tool_result(mem_session_id, "search_worker", text_out[:500])
        log_info("多Agent检索工人", f"执行任务:{search_query}完成，累计结果长度{len(accumulated)}字")

    if need_retry_flag:
        retry_times += 1
        log_info("多Agent调度路由", f"开启检索重试，当前次数={retry_times}，上限={MAX_RETRY}")

    result = {"worker_result": outputs, "need_retry": need_retry_flag, "retry_times": retry_times}
    result.update(extra_fields)
    return result


@global_exception_handler
def lyrics_worker(state: MultiAgentState) -> MultiAgentState:
    """戏词解剖工人：调用 opera.annotate_lyrics"""
    task_list = state["sub_task_list"]
    outputs = list(state["worker_result"])
    mem_session_id = _get_session(state)
    for task in task_list:
        if task["worker"] != "lyrics_worker":
            continue
        lyrics = task.get("params", {}).get("lyrics") or task.get("task", state["user_query"])
        log_info("多Agent戏词工人", f"解剖戏词：{lyrics[:50]}")
        try:
            result = annotate_lyrics(lyrics)
        except Exception as e:
            log_error("多Agent戏词工人", f"戏词解剖失败：{e}", e)
            result = {"error": str(e), "original": lyrics}
        outputs.append({"worker": "lyrics_worker", "task": "戏词解剖", "result": result})
    return {"worker_result": outputs}


@global_exception_handler
def character_worker(state: MultiAgentState) -> MultiAgentState:
    """人物对谈工人：调用 opera.character_chat"""
    task_list = state["sub_task_list"]
    outputs = list(state["worker_result"])
    mem_session_id = _get_session(state)
    history = [m for m in state.get("messages", [])]
    history_ui = []
    for m in history[-8:]:
        role = "user" if isinstance(m, HumanMessage) else "character"
        history_ui.append({"role": role, "content": m.content})
    for task in task_list:
        if task["worker"] != "character_worker":
            continue
        char_id = task.get("params", {}).get("character_id", "muguiying")
        user_msg = task.get("task", state["user_query"])
        log_info("多Agent人物工人", f"角色[{char_id}]对谈：{user_msg[:50]}")
        try:
            result = character_chat(char_id, user_msg, mem_session_id, history_ui)
        except Exception as e:
            log_error("多Agent人物工人", f"对谈失败：{e}", e)
            result = {"error": str(e)}
        if task["worker"] == "character_worker" and "error" not in result:
            # 对谈结束，后续由 summary 包装展示
            result["character_id"] = char_id
        outputs.append({"worker": "character_worker", "task": f"与{char_id}对谈", "result": result})
    return {"worker_result": outputs}


@global_exception_handler
def quiz_worker(state: MultiAgentState) -> MultiAgentState:
    """知识闯关工人：调用 opera.generate_quiz / generate_quiz_batch（动态题量）"""
    from opera import generate_quiz_batch
    task_list = state["sub_task_list"]
    outputs = list(state["worker_result"])
    mem_session_id = _get_session(state)
    intent = state.get("intent") or {}
    for task in task_list:
        if task["worker"] != "quiz_worker":
            continue
        topic = task.get("params", {}).get("topic", "混合")
        difficulty = task.get("params", {}).get("difficulty", "入门")
        # 动态题量：从用户意图/参数解析 count（如"要4道题"→4），默认1道
        count = int(task.get("params", {}).get("count")
                    or task.get("params", {}).get("num")
                    or _extract_quiz_count(state["user_query"])
                    or 1)
        log_info("多Agent闯关工人", f"动态出题：{count}道 {topic} ({difficulty})")
        try:
            if count > 1 or intent.get("reference_material_required"):
                result = generate_quiz_batch(
                    topic=topic, difficulty=difficulty, session_id=mem_session_id,
                    count=count,
                    reference_text=state.get("reference_text", ""),
                    reference_required=bool(intent.get("reference_material_required", False)),
                )
            else:
                result = generate_quiz(topic, difficulty, mem_session_id)
        except Exception as e:
            log_error("多Agent闯关工人", f"出题失败：{e}", e)
            result = {"error": str(e)}
        outputs.append({"worker": "quiz_worker", "task": f"知识闯关出题{count}道", "result": result})
    return {"worker_result": outputs}


def _extract_quiz_count(query: str) -> int:
    """从用户请求中提取题目数量（如'4道题'→4），提取不到返回0"""
    import re as _re
    m = _re.search(r"(\d+)\s*道", query)
    if m:
        return int(m.group(1))
    return 0


@global_exception_handler
def guide_worker(state: MultiAgentState) -> MultiAgentState:
    """学戏路线工人：调用 opera.generate_course"""
    task_list = state["sub_task_list"]
    outputs = list(state["worker_result"])
    mem_session_id = _get_session(state)
    for task in task_list:
        if task["worker"] != "guide_worker":
            continue
        topic = task.get("task", state["user_query"])
        days = int(task.get("params", {}).get("days", 7) or 7)
        level = task.get("params", {}).get("level", "入门")
        log_info("多Agent课程工人", f"生成学戏路线：{topic} {days}天")
        try:
            result = generate_course(topic, days, mem_session_id, level)
        except Exception as e:
            log_error("多Agent课程工人", f"课程生成失败：{e}", e)
            result = {"error": str(e)}
        outputs.append({"worker": "guide_worker", "task": "生成学戏路线", "result": result})
    return {"worker_result": outputs}


@global_exception_handler
def face_worker(state: MultiAgentState) -> MultiAgentState:
    """脸谱画像工人：调用 opera.generate_face_profile（内部含即梦AI图像工具）"""
    task_list = state["sub_task_list"]
    outputs = list(state["worker_result"])
    image_resources = list(state.get("image_resources") or [])
    mem_session_id = _get_session(state)
    for task in task_list:
        if task["worker"] != "face_worker":
            continue
        prefs = task.get("params", {}).get("preferences") or task.get("task", "红色，忠义")
        log_info("多Agent脸谱工人", f"生成脸谱画像：{prefs[:50]}")
        # 修复(Bug)：记录 tool_call 事件（图像工具调用）
        try:
            memory_manager.record_tool_call(mem_session_id, "face_generate_image", prefs[:100])
        except Exception:
            pass
        try:
            result = generate_face_profile(prefs, mem_session_id)
        except Exception as e:
            log_error("多Agent脸谱工人", f"脸谱生成失败：{e}", e)
            result = {"error": str(e)}
        # 修复(Bug2)：规范化图片URL（本地路径→/static URL），补齐 local_file_path/real_file_path
        result = _normalize_face_image_result(result)
        outputs.append({"worker": "face_worker", "task": "生成脸谱画像", "result": result})

        # 修复(Bug)：tool_result 事件 + 图片资源收集
        # 图像工具真实产出图片时，必须产生 tool-result 事件并携带图片路径，
        # 供汇总节点合并进最终返回报文、前端渲染真实图片（禁止只输出文字描述）。
        if "error" not in result:
            image_url = result.get("image_url", "")
            image_status = result.get("image_status", "text_only")
            if image_status == "generated" and image_url:
                img_obj = {
                    "type": "image",
                    "worker": "face_worker",
                    "image_url": image_url,
                    "local_file_path": result.get("local_file_path", ""),
                    "real_file_path": result.get("real_file_path", ""),
                    "face_name": result.get("face_name", ""),
                    "image_status": image_status,
                }
                image_resources.append(img_obj)
                try:
                    memory_manager.record_tool_result(
                        mem_session_id, "face_generate_image",
                        f"image_url={image_url} local_file_path={img_obj['local_file_path']}",
                    )
                    log_info("多Agent脸谱工人", f"图像工具执行完成，tool-result产出图片：{image_url}")
                except Exception:
                    pass
            else:
                try:
                    memory_manager.record_tool_result(
                        mem_session_id, "face_generate_image",
                        "图像生成未产出图片（image_status=" + image_status + "），保留文本版结果",
                    )
                except Exception:
                    pass
    return {"worker_result": outputs, "image_resources": image_resources}


# ===================== 汇总节点 =====================
# 问题3修复：历史对话上下文过滤（复用 intent_guard.filter_history_recent）
#   - 不再把完整 history（含上次AI的闯关题目/文献等输出）注入 prompt
#   - 只提取最近 2 条用户消息作为"对话连贯参考"，且明确标注"仅用于人设/语气，绝不作为生成内容"
from agent.intent_guard import filter_history_recent as _filter_history_recent


@global_exception_handler
def summary_agent(state: MultiAgentState) -> MultiAgentState:
    query = state["user_query"]
    worker_data = state["worker_result"]
    history = state.get("messages", [])
    mem_session_id = _get_session(state)
    mem_task_id = state.get("mem_task_id", None)
    intent = state.get("intent") or {}
    log_info("多Agent汇总节点", "开始整合所有工人结果生成回答")

    # ===== 修复：face_worker 图片资源收集与合并 =====
    # 遍历所有 worker 结果，收集 face_worker 产出的图片资源。
    # 复合需求（如"介绍曹操并生成脸谱"）中 face_worker 是众多 worker 之一，
    # 不能短路径只返回脸谱结果而丢弃其他 worker（如 search_worker）的产出。
    # 策略：
    #   1. 先收集 face_worker 的图片资源（image_resources）
    #   2. 若 face_worker 是唯一 worker → 结构化直出（保留完整图片URL等字段）
    #   3. 若 face_worker 是复合需求的一部分 → 走 LLM 汇总，但强制保留图片资源
    image_resources = list(state.get("image_resources") or [])
    face_result = None
    has_other_workers = False
    for w in worker_data:
        if w.get("worker") == "face_worker" and isinstance(w.get("result"), dict):
            face_res = w["result"]
            if "error" not in face_res:
                face_result = face_res
                # 兜底补充：若 face_worker 已产出真实图片但未收集，这里补上
                image_url = face_res.get("image_url", "")
                if (
                    face_res.get("image_status") == "generated" and image_url
                    and not any(r.get("image_url") == image_url for r in image_resources)
                ):
                    image_resources.append({
                        "type": "image",
                        "worker": "face_worker",
                        "image_url": image_url,
                        "local_file_path": face_res.get("local_file_path", ""),
                        "real_file_path": face_res.get("real_file_path", ""),
                        "face_name": face_res.get("face_name", ""),
                        "image_status": "generated",
                    })
                    log_info(
                        "多Agent汇总节点",
                        f"face_worker 图片资源兜底收集：image_url={image_url}，"
                        f"image_status={face_res.get('image_status')}，"
                        f"face_worker 原始结果 keys={list(face_res.keys())}",
                    )
        elif w.get("worker") != "face_worker":
            has_other_workers = True

    # 场景1：face_worker 是唯一 worker → 结构化直出
    if face_result is not None and not has_other_workers:
        reply_text = _build_face_reply_text(face_result, query)
        log_info(
            "多Agent汇总节点",
            f"face_worker 唯一 worker，结构化直出："
            f"image_url={face_result.get('image_url', '')}，"
            f"image_status={face_result.get('image_status', 'text_only')}，"
            f"reply 长度={len(reply_text)} 字",
        )
        try:
            memory_manager.record_dialogue(mem_session_id, "user", query)
            memory_manager.record_dialogue(mem_session_id, "ai", reply_text[:500])
        except Exception as e:
            log_warn("多Agent汇总节点", f"记录对话轨迹失败：{e}")
        return {
            "messages": [AIMessage(content=reply_text)],
            "image_resources": image_resources,
        }

    # 场景2：face_worker 是复合需求的一部分 → 走 LLM 汇总，强制保留图片资源
    if face_result is not None and has_other_workers:
        log_info(
            "多Agent汇总节点",
            f"复合需求含 face_worker，走 LLM 汇总并强制保留图片资源："
            f"image_resources_count={len(image_resources)}，"
            f"image_url={face_result.get('image_url', '')}",
        )

    # ===== 全新记忆/意图策略 =====
    # 1. 意图对齐：用户当前核心任务最高优先级，forbid 绝对不违反
    # 2. 隔离记忆：仅读取本次任务（mem_task_id）+ 最近会话文本，不跨任务搜索旧输出
    # 3. 历史对话只保留最近用户消息摘要（旧AI工具输出一律不带入）
    intent_prompt_part = ""
    try:
        intent_prompt_part = align_prompt_with_intent(intent)
    except Exception as e:
        log_warn("多Agent汇总节点", f"意图对齐片段生成失败：{e}")
        intent_prompt_part = ""

    mem_context = ""
    try:
        mem_context = get_current_task_only(mem_task_id, session_id=mem_session_id)
    except Exception as e:
        log_warn("多Agent汇总节点", f"隔离记忆上下文检索失败：{e}")
        mem_context = ""

    mem_prompt_part = ""
    if mem_context:
        mem_prompt_part = "附加记忆上下文（仅本次任务，绝不混入旧任务输出）：\n" + mem_context

    # 历史对话过滤：只取最近用户提问摘要，绝不带旧AI工具输出
    history_hint = _filter_history_recent(history)
    history_prompt = ""
    if history_hint:
        history_prompt = (
            "【历史对话连贯参考（仅用于语气/人设，绝不作为本次生成内容依据）】\n"
            + history_hint
        )

    prompt = f"""
{history_prompt}
用户原始提问：{query}

【工人执行结果（本次唯一数据来源）】
{worker_data}
{intent_prompt_part}
{mem_prompt_part}

【汇总要求】
1. 用户当前核心任务优先：严格对齐当前意图，只汇总本次请求对应的功能输出
2. forbid_list 中的行为绝对不得出现
3. 如果用户要求严格基于文档：100%仅基于工人检索的文档内容生成，禁止模型预训练知识/脑补
4. 绝不混入旧任务记忆中的其他功能输出（如上次闯关/上次文献），本次只要单一功能就只输出该功能
5. 历史对话参考仅用于人设/语气连贯，绝不允许把旧任务生成结果（闯关题目/文献/旧检索）作为本次回答内容
6. 如果是戏词解剖（lyrics_worker）：直接输出结构化解读，不要额外评论
7. 如果是人物对谈（character_worker）：以角色口吻回应用户
8. 如果是知识闯关（quiz_worker）：输出题目+选项，提示用户选择
9. 如果是学戏路线（guide_worker）：输出课程大纲
10. 如果是脸谱画像（face_worker）：输出脸谱设计解读+分享文案
11. 如果是知识检索（search_worker）：整合资料完整回答
整合全部信息，给出完整通顺、面向用户的最终回复，不要编造信息。
"""
    try:
        reply = llm.invoke([HumanMessage(content=prompt)])
    except httpx.ConnectError as e:
        err_msg = "汇总节点调用Ollama连接失败"
        log_error("多Agent汇总LLM异常", err_msg, e)
        raise LLMModelException(err_msg, e)
    except Exception as e:
        err_msg = "汇总生成回答流程失败"
        log_error("多Agent汇总流程异常", err_msg, e)
        raise AgentFlowException(err_msg, e)

    try:
        memory_manager.record_dialogue(mem_session_id, "user", query)
        memory_manager.record_dialogue(mem_session_id, "ai", reply.content[:500])
    except Exception as e:
        log_warn("多Agent汇总节点", f"记录对话轨迹失败：{e}")
    log_info(
        "多Agent汇总节点",
        f"最终回答生成完成，image_resources_count={len(image_resources)}",
    )
    # 修复：复合需求中若有 face_worker，必须保留图片资源到返回结果
    return {"messages": [AIMessage(content=reply.content)], "image_resources": image_resources}


# ===================== 路由 =====================
def _get_next_compound_worker(task_arr: list) -> str:
    """获取复合任务列表中下一个要执行的 worker 名称"""
    if not task_arr:
        return "summary_agent"
    w = task_arr[0].get("worker", "")
    if w in ("search_worker", "lyrics_worker", "character_worker", "quiz_worker", "guide_worker", "face_worker"):
        return w
    return "summary_agent"


def _pop_completed_task(task_arr: list) -> list:
    """移除已完成的任务（第一个），返回剩余任务列表"""
    if len(task_arr) <= 1:
        return []
    return task_arr[1:]


def worker_route(state: MultiAgentState) -> Literal[
    "search_worker", "lyrics_worker", "character_worker", "quiz_worker",
    "guide_worker", "face_worker", "summary_agent",
]:
    """按主管分派的任务，路由到对应Worker；无任务直接汇总。
    复合需求时按顺序依次执行各 worker。"""
    task_arr = state["sub_task_list"]
    if not task_arr:
        return "summary_agent"
    w = task_arr[0].get("worker", "")
    if w in ("search_worker", "lyrics_worker", "character_worker", "quiz_worker", "guide_worker", "face_worker"):
        log_info("多Agent调度路由", f"路由到 worker: {w}（剩余 {len(task_arr)-1} 个待执行）")
        return w
    return "summary_agent"


def search_finish_route(state: MultiAgentState) -> Literal["supervisor_node", "summary_agent", "next_worker"]:
    """检索重试判断：统一由 search_worker 的 need_retry 标志控制。
    复合需求时，检索完成后继续执行下一个 worker。
    修复死循环：重试计数由 search_worker 统一递增，此处只做路由决策"""
    need_retry = state.get("need_retry", False)
    retry_times = state.get("retry_times", 0)
    task_arr = state.get("sub_task_list", [])

    if need_retry:
        if retry_times < MAX_RETRY:
            log_info("多Agent调度路由",
                     f"检索结果不足，触发反思重试，当前次数={retry_times}，上限={MAX_RETRY}")
            return "supervisor_node"
        # 达到最大重试上限，强制终止
        log_warn("多Agent调度路由",
                 f"检索重试次数已达上限{MAX_RETRY}，强制终止重试，使用当前已有结果进入汇总")
        return "summary_agent"

    # 复合需求：检索完成后检查是否还有下一个 worker
    remaining = _pop_completed_task(task_arr)
    if remaining:
        log_info("多Agent调度路由", f"复合需求：检索完成，继续执行下一个 worker：{remaining[0].get('worker')}")
        return "next_worker"
    log_info("多Agent调度路由", "检索结果充足，进入汇总")
    return "summary_agent"


def next_worker_route(state: MultiAgentState) -> Literal[
    "search_worker", "lyrics_worker", "character_worker", "quiz_worker",
    "guide_worker", "face_worker", "summary_agent",
]:
    """复合需求：非检索 worker 完成后，路由到下一个 worker。
    通过检查 sub_task_list 中剩余的 worker 来决定下一步。"""
    task_arr = state.get("sub_task_list", [])
    remaining = _pop_completed_task(task_arr)
    if remaining:
        next_w = remaining[0].get("worker", "")
        if next_w in ("search_worker", "lyrics_worker", "character_worker", "quiz_worker", "guide_worker", "face_worker"):
            log_info("多Agent调度路由", f"复合需求：继续执行下一个 worker：{next_w}（剩余 {len(remaining)-1} 个）")
            return next_w
    log_info("多Agent调度路由", "复合需求：所有 worker 执行完毕，进入汇总")
    return "summary_agent"


# ===================== 复合需求多Worker调度节点 =====================
def _compound_worker_finish(state: MultiAgentState) -> dict:
    """复合需求：一个 worker 完成后，从 sub_task_list 中移除已完成的任务。
    返回更新后的 sub_task_list，供 next_worker_route 路由到下一个 worker。"""
    task_arr = list(state.get("sub_task_list", []))
    remaining = _pop_completed_task(task_arr)
    log_info("复合需求调度", f"Worker 完成，剩余待执行：{remaining}")
    return {"sub_task_list": remaining}


def build_multi_agent():
    log_info("多Agent构建", "初始化主管-工人多智能体流程图（支持复合需求多Worker串行调度）")
    graph = StateGraph(MultiAgentState)
    graph.add_node("supervisor_node", supervisor_node)
    graph.add_node("search_worker", search_worker)
    graph.add_node("lyrics_worker", lyrics_worker)
    graph.add_node("character_worker", character_worker)
    graph.add_node("quiz_worker", quiz_worker)
    graph.add_node("guide_worker", guide_worker)
    graph.add_node("face_worker", face_worker)
    graph.add_node("summary_agent", summary_agent)
    # 复合需求多Worker调度节点：在每个非检索 worker 完成后，移除已完成任务并路由到下一个
    graph.add_node("compound_worker_finish", _compound_worker_finish)
    graph.set_entry_point("supervisor_node")

    # supervisor → 按工人路由
    graph.add_conditional_edges("supervisor_node", worker_route, {
        "search_worker": "search_worker",
        "lyrics_worker": "lyrics_worker",
        "character_worker": "character_worker",
        "quiz_worker": "quiz_worker",
        "guide_worker": "guide_worker",
        "face_worker": "face_worker",
        "summary_agent": "summary_agent",
    })

    # 检索工人 → 重试判断 → supervisor 或 next_worker 或 汇总
    graph.add_conditional_edges("search_worker", search_finish_route, {
        "supervisor_node": "supervisor_node",
        "next_worker": "compound_worker_finish",
        "summary_agent": "summary_agent",
    })

    # 非检索工人 → compound_worker_finish → next_worker_route 判断
    for node in ("lyrics_worker", "character_worker", "quiz_worker", "guide_worker", "face_worker"):
        graph.add_edge(node, "compound_worker_finish")

    # compound_worker_finish → 根据剩余任务数路由到下一个 worker 或汇总
    graph.add_conditional_edges("compound_worker_finish", next_worker_route, {
        "search_worker": "search_worker",
        "lyrics_worker": "lyrics_worker",
        "character_worker": "character_worker",
        "quiz_worker": "quiz_worker",
        "guide_worker": "guide_worker",
        "face_worker": "face_worker",
        "summary_agent": "summary_agent",
    })

    graph.add_edge("summary_agent", END)
    log_info("多Agent构建", "多智能体流程图构建完成（复合需求支持）")
    return graph.compile()


if __name__ == "__main__":
    agent = build_multi_agent()
    print("主管-工人多智能体（角色化，带检索重试）已经启动")
    while True:
        inp = input("用户：")
        if inp == "exit":
            break
        init = {
            "user_query": inp,
            "sub_task_list": [],
            "worker_result": [],
            "messages": [],
            "retry_times": 0,
            "need_retry": False,
            "mem_session_id": "interactive",
            "mem_task_id": None,
            "intent": {},
            "reference_text": "",
            "image_resources": [],
            "blackboard": {},
        }
        try:
            result = agent.invoke(init)
            print(f"AI回答：{result['messages'][-1].content}\n")
        except Exception as e:
            log_error("多Agent交互脚本", f"执行失败：{e}", e)
            print(f"执行失败：{e}")