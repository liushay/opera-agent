from typing import TypedDict, Annotated, Sequence, Literal
import operator
import json
import httpx
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
import config
from tools.custom_tools import tool_list, knowledge_tool, calc_tool
from utils.logger import log_debug, log_info, log_warn, log_error
from utils.rag_exceptions import LLMModelException, AgentFlowException, VectorStoreException, BM25IndexException
from utils.exception_handler import global_exception_handler
from agent.memory import memory_manager

# 默认会话ID（用于时序记忆记录），外部可通过state传入
DEFAULT_MEM_SESSION_ID = config.DEFAULT_SESSION_ID
# 当前任务ID（可选用），用于任务级记忆
DEFAULT_MEM_TASK_ID = None

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
@global_exception_handler
def plan_tool_call(state: AgentState) -> AgentState:
    log_info("Agent规划节点", f"开始规划工具调用，用户问题：{state['user_query']}")
    # 检索分层记忆上下文（可选字段，兼容旧调用方）
    mem_session_id = state.get("mem_session_id", DEFAULT_MEM_SESSION_ID)
    mem_task_id = state.get("mem_task_id", DEFAULT_MEM_TASK_ID)
    try:
        mem_context = memory_manager.format_memory_context(
            query=state["user_query"],
            session_id=mem_session_id,
            task_id=mem_task_id,
        )
        # 记录时序事件
        memory_manager.record_thinking(mem_session_id, f"规划阶段：用户提问{state['user_query']}")
    except Exception as e:
        log_warn("Agent规划节点", f"记忆上下文检索失败，降级为无记忆：{e}")
        mem_context = ""

    prompt = f"""
可用工具列表：
{tools_info}
规则：
1. RAG/分块/向量库技术问题：调用search_knowledge_base
2. 数字四则运算：调用calculator
3. 日常闲聊、无计算无查询：{{"tool_name":"none"}}
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
    try:
        if tool_name == "search_knowledge_base":
            res = knowledge_tool.invoke(params)
        elif tool_name == "calculator":
            res = calc_tool.invoke(params)
    except (VectorStoreException, BM25IndexException) as e:
        err_msg = f"工具{tool_name}底层检索失败"
        log_error("Agent工具执行异常", err_msg, e)
        raise AgentFlowException(err_msg, e)
    except Exception as e:
        err_msg = f"工具{tool_name}执行未知错误"
        log_error("Agent工具执行异常", err_msg, e)
        raise AgentFlowException(err_msg, e)
    # 记录工具结果（记忆增强）
    try:
        memory_manager.record_tool_result(mem_session_id, tool_name, str(res)[:500])
    except Exception as e:
        log_warn("Agent工具执行节点", f"记录工具结果失败：{e}")
    log_info("Agent工具执行节点", f"工具{tool_name}执行完成")
    return {"tool_result": res}

# 节点3：反思校验
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

    prompt = f"""
用户问题：{q}
当前工具返回资料：{tool_res}
判断规则：仅输出 True / False
True = 现有信息足够完整回答用户问题
False = 信息缺失，需要重新检索知识库补充内容
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

    flag = resp.content.strip() == "True"
    if not flag:
        log_info("Agent反思", "信息不足，需要重新检索")
        return {"reflect_times": reflect_cnt + 1}
    log_info("Agent反思", "信息充足，直接生成答案")
    return {"reflect_times": reflect_cnt}

# 节点4：生成最终回答
@global_exception_handler
def generate_final_ans(state: AgentState) -> AgentState:
    history = state["messages"]
    q = state["user_query"]
    tool_res = state["tool_result"]
    log_info("Agent生成回答节点", "开始整合信息生成最终回复")
    # 检索记忆上下文（增强提示词）
    mem_session_id = state.get("mem_session_id", DEFAULT_MEM_SESSION_ID)
    mem_task_id = state.get("mem_task_id", DEFAULT_MEM_TASK_ID)
    try:
        mem_context = memory_manager.format_memory_context(
            query=q,
            session_id=mem_session_id,
            task_id=mem_task_id,
        )
    except Exception as e:
        log_warn("Agent生成回答节点", f"记忆上下文检索失败：{e}")
        mem_context = ""

    # 提前拼接记忆上下文，避免f-string反斜杠语法错误
    mem_prompt_part = ""
    if mem_context:
        mem_prompt_part = "附加记忆上下文：\n" + mem_context
    prompt = f"""
历史对话上下文：{history}
用户当前提问：{q}
工具查询参考资料：{tool_res}
{mem_prompt_part}
要求：严格依据提供资料回答，禁止编造未出现信息，回答简洁通顺。
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
    q = state["user_query"]
    tool_res = state["tool_result"]
    reflect_cnt = state["reflect_times"]
    max_reflect = config.MAX_REFLECT_TIMES
    if reflect_cnt >= max_reflect:
        return "generate_final_ans"
    prompt = f"问题：{q}，现有资料：{tool_res}，信息是否充足？只输出True/False"
    try:
        res = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception:
        log_warn("Agent反思路由", "反思判断LLM调用失败，直接生成答案")
        return "generate_final_ans"
    if res == "False":
        return "plan_tool_call"
    else:
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
    log_info("Agent构建", "单智能体流程图构建完成")
    return graph.compile()