from typing import TypedDict, Annotated, Sequence, Literal
import operator
import json
import httpx
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
import config
from tools.custom_tools import tool_list, knowledge_tool
from utils.logger import log_debug, log_info, log_warn, log_error
from utils.rag_exceptions import LLMModelException, AgentFlowException, VectorStoreException, BM25IndexException
from utils.exception_handler import global_exception_handler
from agent.memory import memory_manager
from agent.trace import agent_tracer
from agent.intent import align_prompt_with_intent
from agent.task_context import get_current_task_only
import time as _time

# 默认会话ID（用于时序记忆记录），外部可通过state传入
DEFAULT_MEM_SESSION_ID = config.DEFAULT_SESSION_ID
# 当前任务ID（可选用），用于任务级记忆
DEFAULT_MEM_TASK_ID = None

# 当前活跃 trace_id（由 build_agent_graph 中包装）
# 通过简单的 thread-local 避免修改所有节点签名
import threading
_trace_local = threading.local()


def set_trace(trace_id: str):
    """设置当前线程的 trace_id（由 invoke 包装传入）"""
    _trace_local.trace_id = trace_id


def get_trace() -> str:
    """获取当前线程的 trace_id"""
    return getattr(_trace_local, "trace_id", "")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    tool_call: dict
    tool_result: str
    reflect_times: int
    # 记忆增强字段（可选，不影响原有调用方）
    mem_session_id: str
    mem_task_id: str
    mem_context: str
    # 意图解析结果（新增，可选字段兼容旧调用方）
    intent: dict
    reference_text: str
    # ===== 反思增强字段（差异化反思 + 反馈驱动重规划） =====
    last_reflection: dict       # 上次反思的结构化诊断报告 {sufficient, missing_aspects, suggested_queries, reason}
    last_query: str             # 上次检索使用的 query，用于禁止重复
    accumulated_results: str    # 多轮检索结果累积拼接，而非覆盖

# 初始化LLM
def get_llm():
    try:
        return ChatOllama(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMP
        )
    except httpx.ConnectError as e:
        err_msg = f"Agent初始化连接Ollama失败，模型{config.LLM_MODEL}"
        log_error("Agent LLM初始化失败", err_msg, e)
        raise LLMModelException(err_msg, e)
    except Exception as e:
        err_msg = f"Agent LLM实例创建失败"
        log_error("Agent LLM异常", err_msg, e)
        raise LLMModelException(err_msg, e)

llm = get_llm()
tools_info = "\n".join([f"工具名称：{t.name}，功能：{t.description}" for t in tool_list])

# 节点1：工具规划
# 问题3修复：
#   1. 使用任务隔离记忆（get_current_task_only），不再跨任务检索旧工具结果
#   2. 接入意图解析结果（intent），当前用户核心任务 = 最高优先级
#   3. 知识问答/介绍类请求强制走检索；日常闲聊直接 none
def _should_call_search(state: AgentState) -> bool:
    """意图校验：判断用户本轮是否需要调用检索工具"""
    query = state.get("user_query", "").strip()
    if not query:
        return False
    intent = state.get("intent") or {}
    # 用户明确要求基于文档/知识库时，必须检索
    if intent.get("reference_material_required"):
        return True
    # 日常闲聊不检索
    chat_keywords = ("你好", "谢谢", "再见", "你是谁", "你能做什么", "hi", "hello", "无聊", "开心")
    if any(kw in query.lower() for kw in chat_keywords):
        return False
    return True


@global_exception_handler
def plan_tool_call(state: AgentState) -> AgentState:
    log_info("Agent规划节点", f"开始规划工具调用，用户问题：{state['user_query']}")
    mem_session_id = state.get("mem_session_id", DEFAULT_MEM_SESSION_ID)
    mem_task_id = state.get("mem_task_id", DEFAULT_MEM_TASK_ID)
    # 问题3修复：使用任务隔离记忆（仅本次任务 + 最近对话，绝不含旧任务工具输出）
    try:
        mem_context = get_current_task_only(mem_task_id, session_id=mem_session_id)
        # 记录时序事件
        memory_manager.record_thinking(mem_session_id, f"规划阶段：用户提问{state['user_query']}")
    except Exception as e:
        log_warn("Agent规划节点", f"记忆上下文检索失败，降级为无记忆：{e}")
        mem_context = ""

    # 意图约束片段（用户当前核心任务最高优先级）
    intent = state.get("intent") or {}
    intent_constraint = ""
    if intent.get("core_task"):
        intent_constraint = f"\n用户本次核心任务：{intent.get('core_task', '')}\n只执行该核心任务相关操作，绝不追加额外任务。"
    if intent.get("forbid_list"):
        intent_constraint += "\n禁止行为：" + "；".join(f"- {f}" for f in intent["forbid_list"])

    # Trace 记录
    trace_id = get_trace()
    if trace_id:
        agent_tracer.add_step(trace_id, "plan_tool_call", f"规划工具调用")

    # ===== 反思增强：注入上次反思反馈，驱动差异化决策 =====
    last_reflection = state.get("last_reflection", {})
    last_query = state.get("last_query", "")
    reflect_cnt = state.get("reflect_times", 0)
    reflection_feedback = ""
    if last_reflection and reflect_cnt > 0:
        missing = last_reflection.get("missing_aspects", [])
        suggested = last_reflection.get("suggested_queries", [])
        reason = last_reflection.get("reason", "")
        reflection_feedback = f"""
【上一轮反思诊断报告——请务必据此调整检索策略】
- 上次检索结果不足的原因：{reason}
- 缺失的信息维度：{', '.join(missing) if missing else '无'}
- 建议尝试的检索关键词：{', '.join(suggested) if suggested else '无'}
- 上次使用的检索query（禁止重复使用）："{last_query}"

【硬约束——检索主题必须锚定用户原始问题】
用户原始问题："{state['user_query']}"
新检索query必须满足：
1. 必须包含用户原始问题的核心主题词
2. 只能在此基础上增加细节、角度、具体化，不能改变主题
3. 禁止将query替换为完全不同的主题
4. 禁止使用与上次完全相同的检索query
5. 可以尝试：拆分查询、换同义词、从具体人物/剧目角度切入
"""
    # ===== 反思增强结束 =====

    prompt = f"""
可用工具列表：
{tools_info}
{intent_constraint}
{reflection_feedback}
规则：
1. 用户核心任务是知识问答/检索/介绍/解释类：调用search_knowledge_base
2. 日常闲聊、无检索需求：{{"tool_name":"none"}}
3. 只允许调用用户当前问题需要的工具，绝不自动追加用户未提出的任务
输出要求：仅输出纯JSON，无任何多余文字
格式示例：{{"tool_name":"search_knowledge_base","params":{{"query":"xxx"}}}}
用户问题：{state['user_query']}
"""
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
    except httpx.ConnectError as e:
        err_msg = "规划节点调用Ollama连接失败"
        log_error("Agent规划LLM异常", err_msg, e)
        raise LLMModelException(err_msg, e)
    except httpx.TimeoutException as e:
        err_msg = "规划节点Ollama调用超时"
        log_error("Agent规划LLM超时", err_msg, e)
        raise LLMModelException(err_msg, e)
    except Exception as e:
        err_msg = "规划节点模型生成失败"
        log_error("Agent规划流程异常", err_msg, e)
        raise AgentFlowException(err_msg, e)

    try:
        call_data = json.loads(resp.content.strip())
    except json.JSONDecodeError as e:
        log_warn("Agent规划", "模型输出非标准JSON，工具调用设为none", e)
        call_data = {"tool_name": "none", "params": {}}

    # ===== 意图校验：无检索需求时强制 none =====
    if not _should_call_search(state):
        call_data = {"tool_name": "none", "params": {}}
        log_info("Agent规划节点", "意图校验：日常闲聊/无检索需求，不调用工具")

    log_info("Agent规划节点", f"规划完成，待调用工具：{call_data['tool_name']}")
    return {"tool_call": call_data}

# 节点2：执行工具
@global_exception_handler
def run_tool(state: AgentState) -> AgentState:
    call_info = state["tool_call"]
    tool_name = call_info["tool_name"]
    params = call_info["params"]
    res = "无需调用任何工具"
    log_info("Agent工具执行节点", f"执行工具：{tool_name}")
    # 记录工具调用（记忆增强，可选字段兼容旧调用方）
    mem_session_id = state.get("mem_session_id", DEFAULT_MEM_SESSION_ID)
    try:
        memory_manager.record_tool_call(mem_session_id, tool_name, str(params))
    except Exception as e:
        log_warn("Agent工具执行节点", f"记录工具调用失败：{e}")
    # Trace 记录工具调用（开始计时）
    trace_id = get_trace()
    _tool_start = _time.time()
    try:
        if tool_name == "search_knowledge_base":
            res = knowledge_tool.invoke(params)
    except (VectorStoreException, BM25IndexException) as e:
        if trace_id:
            agent_tracer.record_tool_call(
                trace_id, tool_name, params, str(e), 
                (_time.time() - _tool_start) * 1000, False,
            )
        err_msg = f"工具{tool_name}底层检索失败"
        log_error("Agent工具执行异常", err_msg, e)
        raise AgentFlowException(err_msg, e)
    except Exception as e:
        if trace_id:
            agent_tracer.record_tool_call(
                trace_id, tool_name, params, str(e),
                (_time.time() - _tool_start) * 1000, False,
            )
        err_msg = f"工具{tool_name}执行未知错误"
        log_error("Agent工具执行异常", err_msg, e)
        raise AgentFlowException(err_msg, e)
    # Trace 记录工具调用（成功）
    if trace_id:
        agent_tracer.record_tool_call(
            trace_id, tool_name, params, str(res)[:200],
            (_time.time() - _tool_start) * 1000, True,
        )
    # 记录工具结果（记忆增强）
    try:
        memory_manager.record_tool_result(mem_session_id, tool_name, str(res)[:500])
    except Exception as e:
        log_warn("Agent工具执行节点", f"记录工具结果失败：{e}")

    # ===== 反思增强：记录 last_query + 累积检索结果 =====
    reflect_cnt = state.get("reflect_times", 0)
    update_data = {"tool_result": res}
    if tool_name == "search_knowledge_base":
        # 记录本次检索 query，供下一轮反思禁止重复使用
        current_query = params.get("query", "")
        update_data["last_query"] = current_query
        # 重试时累积结果而非覆盖，让信息池不断增长
        if reflect_cnt > 0:
            previous = state.get("tool_result", "")
            accumulated = state.get("accumulated_results", previous)
            if accumulated:
                accumulated = accumulated + f"\n\n--- 补充检索结果（第{reflect_cnt}次重试，query: {current_query}）---\n{res}"
            else:
                accumulated = res
            update_data["tool_result"] = accumulated
            update_data["accumulated_results"] = accumulated
            log_info("Agent工具执行节点", f"累积检索结果，当前总长度{len(accumulated)}字")
        else:
            update_data["accumulated_results"] = res
    # ===== 反思增强结束 =====

    log_info("Agent工具执行节点", f"工具{tool_name}执行完成")
    return update_data

# 节点3：反思校验（增强版：输出结构化诊断报告）
@global_exception_handler
def reflect_check(state: AgentState) -> AgentState:
    q = state["user_query"]
    tool_res = state["tool_result"]
    reflect_cnt = state["reflect_times"]
    max_reflect = config.MAX_REFLECT_TIMES
    log_info("Agent反思节点", f"当前反思次数：{reflect_cnt}/{max_reflect}")
    # 记录反思阶段（记忆增强）
    mem_session_id = state.get("mem_session_id", DEFAULT_MEM_SESSION_ID)
    try:
        memory_manager.record_thinking(
            mem_session_id,
            f"反思阶段：第{reflect_cnt}次反思，判断信息是否充足",
        )
    except Exception as e:
        log_warn("Agent反思节点", f"记录反思记忆失败：{e}")
    if reflect_cnt >= max_reflect:
        log_warn("Agent反思", "已达到最大反思上限，停止重新检索")
        return {"reflect_times": reflect_cnt}

    # ===== 反思增强：输出结构化诊断报告 =====
    prompt = f"""
用户原始问题（唯一参照标准）：{q}
当前检索到的资料：{tool_res}
当前反思次数：第{reflect_cnt + 1}次（共{max_reflect}次）

请以"用户原始问题"为唯一参照标准，判断现有资料是否足够完整回答。输出严格JSON（不要输出任何其他内容）：
{{
  "sufficient": true或false,
  "missing_aspects": ["相对于用户原始问题，还缺少的信息维度1", "维度2"],
  "suggested_queries": ["建议的检索关键词1（必须保留用户原始问题的核心主题词）", "关键词2"],
  "reason": "为什么不足/为什么足够（必须明确说明与用户原始问题之间的差距）"
}}

【判断标准】
- 如果资料已经覆盖了用户原始问题的所有关键点 → sufficient=true
- 如果资料不相关、不完整、或只覆盖了部分问题 → sufficient=false
- 如果这是第2次及以上反思（reflect_cnt>=1），suggested_queries 必须给出与之前不同的检索方向
- suggested_queries 中的每个query必须包含用户原始问题的核心主题词，禁止偏离主题

【输出规范】
- 如果 sufficient=true，missing_aspects 和 suggested_queries 可以为空数组
- 如果 sufficient=false，missing_aspects 和 suggested_queries 必须至少包含1个元素
"""
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
    except httpx.ConnectError as e:
        err_msg = "反思节点Ollama连接失败"
        log_error("Agent反思LLM异常", err_msg, e)
        raise LLMModelException(err_msg, e)
    except Exception as e:
        err_msg = "反思节点执行失败"
        log_error("Agent反思流程异常", err_msg, e)
        raise AgentFlowException(err_msg, e)

    # 解析结构化诊断报告
    try:
        reflection = json.loads(resp.content.strip())
    except json.JSONDecodeError:
        log_warn("Agent反思", "反思输出非标准JSON，降级为传统True/False判断")
        flag = resp.content.strip() == "True"
        reflection = {
            "sufficient": flag,
            "missing_aspects": [],
            "suggested_queries": [],
            "reason": "反思JSON解析失败，降级为传统判断",
        }

    sufficient = reflection.get("sufficient", False)
    if not sufficient:
        log_info("Agent反思",
                 f"信息不足，需要重新检索。缺失维度：{reflection.get('missing_aspects', [])}，"
                 f"建议关键词：{reflection.get('suggested_queries', [])}")
        return {
            "reflect_times": reflect_cnt + 1,
            "last_reflection": reflection,
        }
    log_info("Agent反思", "信息充足，直接生成答案")
    return {
        "reflect_times": reflect_cnt,
        "last_reflection": reflection,
    }
    # ===== 反思增强结束 =====

# 节点4：生成最终回答
def _filter_history_hint(history, max_user_msgs: int = 2) -> str:
    """
    问题3修复：历史对话只提取最近用户消息摘要，过滤掉旧AI工具输出。
    绝不让上一次的闯关题目/文献/检索结果混入本次生成上下文。
    """
    if not history:
        return ""
    user_msgs = []
    for m in history:
        if isinstance(m, HumanMessage):
            user_msgs.append(m.content)
    if not user_msgs:
        return ""
    recent = user_msgs[-max_user_msgs:]
    return "\n".join(f"- 用户之前问过：{content[:100]}" for content in recent)


@global_exception_handler
def generate_final_ans(state: AgentState) -> AgentState:
    history = state["messages"]
    q = state["user_query"]
    tool_res = state["tool_result"]
    log_info("Agent生成回答节点", "开始整合信息生成最终回复")
    # ===== 记忆增强（全新隔离策略） =====
    # 1. 优先使用意图解析结果（core_task/forbid/文档约束） —— 用户当前提问 = 最高优先级
    # 2. 记忆只做辅助：仅读取本次任务（mem_task_id）的隔离上下文，
    #    绝不跨任务搜索旧任务输出，绝不混入旧任务的闯关/文献/问答产物
    # 3. 历史对话只保留最近用户消息摘要，旧AI工具输出一律不带入
    mem_session_id = state.get("mem_session_id", DEFAULT_MEM_SESSION_ID)
    mem_task_id = state.get("mem_task_id", DEFAULT_MEM_TASK_ID)
    intent = state.get("intent") or {}

    # 意图对齐片段（核心：core_task / must / forbid / reference_material_required）
    intent_prompt_part = ""
    try:
        intent_prompt_part = align_prompt_with_intent(intent)
    except Exception as e:
        log_warn("Agent生成回答节点", f"意图对齐片段生成失败：{e}")
        intent_prompt_part = ""

    # 隔离记忆上下文（仅本次任务 + 最近会话文本，不跨任务）
    mem_context = ""
    try:
        mem_context = get_current_task_only(mem_task_id, session_id=mem_session_id)
    except Exception as e:
        log_warn("Agent生成回答节点", f"隔离记忆上下文检索失败：{e}")
        mem_context = ""

    # 参考文档文本（当用户要求严格基于文档时注入）
    reference_text = state.get("reference_text", "")

    # 提前拼接记忆上下文，避免f-string反斜杠语法错误
    mem_prompt_part = ""
    if mem_context:
        mem_prompt_part = "附加记忆上下文（仅本次任务，绝不混入旧任务输出）：\n" + mem_context

    # 历史对话过滤（问题3修复）
    history_hint = _filter_history_hint(history)
    history_prompt = ""
    if history_hint:
        history_prompt = (
            "【历史对话连贯参考（仅用于语气/人设，绝不作为本次生成内容依据）】\n"
            + history_hint
        )

    reference_prompt_part = ""
    if intent.get("reference_material_required"):
        reference_prompt_part = (
            "\n【参考文档（必须严格基于此内容生成，禁止使用模型预训练知识/外部知识点/脑补）】\n"
            + (reference_text[:4000] if reference_text else "（无额外参考文档，仅能基于工具检索结果）")
        )

    prompt = f"""
{history_prompt}
用户当前提问：{q}
工具查询参考资料：{tool_res}
{reference_prompt_part}
{intent_prompt_part}
{mem_prompt_part}
【生成要求】
1. 用户当前核心任务优先于一切历史记忆，严格对齐当前意图
2. 只做本次任务要求的事，绝不擅自追加旧任务或其他功能内容
3. 若用户要求严格基于文档：100%仅基于参考文档生成，禁止模型预训练知识、禁止脑补
4. 若用户明确禁止某些行为：绝对不得违反 forbid_list
5. 历史对话参考仅用于语气/人设连贯，绝不允许把旧任务生成结果（闯关题目/文献/旧检索）作为本次回答内容
6. 回答简洁通顺，直接回应用户当前问题
"""
    try:
        ans = llm.invoke([HumanMessage(content=prompt)])
    except httpx.ConnectError as e:
        err_msg = "生成回答节点Ollama连接失败"
        log_error("Agent生成LLM异常", err_msg, e)
        raise LLMModelException(err_msg, e)
    except Exception as e:
        err_msg = "生成最终回答流程失败"
        log_error("Agent生成流程异常", err_msg, e)
        raise AgentFlowException(err_msg, e)
    # 记录对话轨迹（记忆增强）
    try:
        memory_manager.record_dialogue(mem_session_id, "user", q)
        memory_manager.record_dialogue(mem_session_id, "ai", ans.content[:500])
    except Exception as e:
        log_warn("Agent生成回答节点", f"记录对话轨迹失败：{e}")
    log_info("Agent生成回答节点", "AI回复生成完成")
    return {"messages": [AIMessage(content=ans.content)]}

# 路由
def route_by_tool(state: AgentState) -> Literal["run_tool", "generate_final_ans"]:
    if state["tool_call"]["tool_name"] == "none":
        return "generate_final_ans"
    return "run_tool"

def route_reflect(state: AgentState) -> Literal["plan_tool_call", "generate_final_ans"]:
    """路由：根据 reflect_check 产出的结构化诊断报告决定是否重试"""
    reflect_cnt = state["reflect_times"]
    max_reflect = config.MAX_REFLECT_TIMES
    if reflect_cnt >= max_reflect:
        log_info("Agent反思路由", f"反思次数已达上限{max_reflect}，直接生成答案")
        return "generate_final_ans"
    # 优先使用结构化诊断报告中的 sufficient 字段
    last_reflection = state.get("last_reflection", {})
    if last_reflection.get("sufficient") is False:
        log_info("Agent反思路由", "诊断报告判定信息不足，返回规划节点重新检索")
        return "plan_tool_call"
    # 降级：如果诊断报告缺失，传统方式判断
    if last_reflection:
        log_info("Agent反思路由", "诊断报告判定信息充足，生成最终答案")
        return "generate_final_ans"
    # 完全降级（无诊断报告时，保守起见直接生成答案）
    log_warn("Agent反思路由", "无诊断报告，降级直接生成答案")
    return "generate_final_ans"

def build_agent_graph():
    log_info("Agent构建", "初始化单智能体流程图")
    graph = StateGraph(AgentState)
    graph.add_node("plan_tool_call", plan_tool_call)
    graph.add_node("run_tool", run_tool)
    graph.add_node("reflect_check", reflect_check)
    graph.add_node("generate_final_ans", generate_final_ans)
    graph.set_entry_point("plan_tool_call")
    graph.add_conditional_edges(
        source="plan_tool_call",
        path=route_by_tool,
        path_map={
            "run_tool": "run_tool",
            "generate_final_ans": "generate_final_ans"
        }
    )
    graph.add_edge("run_tool", "reflect_check")
    graph.add_conditional_edges(
        source="reflect_check",
        path=route_reflect,
        path_map={
            "plan_tool_call": "plan_tool_call",
            "generate_final_ans": "generate_final_ans"
        }
    )
    graph.add_edge("generate_final_ans", END)
    compiled = graph.compile()

    # 包装 invoke：自动开启/结束 Trace（不改变返回结构）
    orig_invoke = compiled.invoke

    def traced_invoke(state: dict, *args, **kwargs):
        session_id = state.get("mem_session_id", DEFAULT_MEM_SESSION_ID)
        user_query = state.get("user_query", "")
        trace_id = agent_tracer.start_trace(
            name="single_agent",
            session_id=str(session_id),
            user_query=str(user_query),
        )
        set_trace(trace_id)
        try:
            result = orig_invoke(state, *args, **kwargs)
            agent_tracer.end_trace(trace_id, status="success", result=result)
            return result
        except Exception as e:
            agent_tracer.end_trace(trace_id, status="error", error=str(e))
            raise
        finally:
            _trace_local.trace_id = None

    compiled.invoke = traced_invoke
    log_info("Agent构建", "单智能体流程图构建完成（已启用Trace追踪）")
    return compiled
