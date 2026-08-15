from typing import TypedDict, Annotated, Sequence, Literal
import operator
import json
import httpx

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama

import config
from tools.custom_tools import calculator, search_knowledge_base
from tools.tool_file_query import read_local_file
from utils.logger import log_debug, log_info, log_warn, log_error
from utils.rag_exceptions import LLMModelException, AgentFlowException, VectorStoreException, BM25IndexException
from utils.exception_handler import global_exception_handler

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    tool_call: dict
    tool_result: str
    reflect_times: int

# ----------------------工具&LLM初始化----------------------
tool_list = [calculator, search_knowledge_base, read_local_file]

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
llm_with_tools = llm.bind_tools(tool_list)

# ----------------------新增节点----------------------
@global_exception_handler
def agent_node(state: AgentState) -> AgentState:
    log_info("Agent‑自主规划节点", f"用户问题:{state['user_query']}")
    # Ollama要求消息列表非空：若messages为空则用用户查询作为初始消息
    if not state["messages"]:
        log_warn("Agent规划节点", "消息列表为空，使用用户查询作为初始消息")
        messages = [HumanMessage(content=state["user_query"])]
    else:
        messages = state["messages"]
    try:
        resp = llm_with_tools.invoke(messages)
    except ValueError as e:
        if "No data received from Ollama stream" in str(e):
            log_warn("Agent规划节点", f"Ollama流式返回空数据，重试一次，原始错误：{e}")
            resp = llm_with_tools.invoke(messages)
        else:
            raise
    return {"messages": [resp]}


@global_exception_handler
def sync_tool_result(state: AgentState) -> AgentState:
    """中转节点：把ToolNode返回的ToolMessage内容同步存入tool_result，兼容原有反思逻辑"""
    last_msg = state["messages"][-1]
    tool_content = getattr(last_msg, "content", "")
    log_info("工具结果同步节点", "已将工具返回内容写入tool_result")
    return {"tool_result": tool_content}


# ----------------------原有节点【完全不动，原样保留】----------------------
@global_exception_handler
def reflect_check(state: AgentState) -> AgentState:
    q = state["user_query"]
    tool_res = state["tool_result"]
    reflect_cnt = state["reflect_times"]
    max_reflect = config.MAX_REFLECT_TIMES
    log_info("Agent反思节点", f"当前反思次数：{reflect_cnt}/{max_reflect}")
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


@global_exception_handler
def generate_final_ans(state: AgentState) -> AgentState:
    history = state["messages"]
    q = state["user_query"]
    tool_res = state["tool_result"]
    log_info("Agent生成回答节点", "开始整合信息生成最终回复")
    prompt = f"""
历史对话上下文：{history}
用户当前提问：{q}
工具查询参考资料：{tool_res}
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
    log_info("Agent生成回答节点", "AI回复生成完成")
    return {"messages": [AIMessage(content=ans.content)]}


# ----------------------路由函数----------------------
def route_agent_output(state: AgentState) -> Literal["tools", "reflect_check"]:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
        return "tools"
    return "reflect_check"


def route_reflect(state: AgentState) -> Literal["agent_node", "generate_final_ans"]:
    q = state["user_query"]
    tool_res = state["tool_result"]
    reflect_cnt = state["reflect_times"]
    max_reflect = config.MAX_REFLECT_TIMES
    if reflect_cnt >= max_reflect:
        log_info("反思路由", "达到最大重试次数，直接生成答案")
        return "generate_final_ans"
    prompt = f"问题：{q}，现有资料：{tool_res}，信息是否充足？只输出True/False"
    try:
        res = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception:
        log_warn("Agent反思路由", "反思判断LLM调用失败，直接生成答案")
        return "generate_final_ans"
    if res == "False":
        log_info("反思路由", "信息不足，回流agent重新检索")
        return "agent_node"
    else:
        return "generate_final_ans"


# ----------------------构建流程图----------------------
def build_agent_graph():
    log_info("Agent构建", "初始化bind‑tools工具调用流程图")
    tool_node = ToolNode(tool_list)
    graph = StateGraph(AgentState)

    # 注册全部节点
    graph.add_node("agent_node", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("sync_tool_result", sync_tool_result)
    graph.add_node("reflect_check", reflect_check)
    graph.add_node("generate_final_ans", generate_final_ans)

    graph.set_entry_point("agent_node")

    # agent分支路由
    graph.add_conditional_edges(
        source="agent_node",
        path=route_agent_output,
        path_map={
            "tools": "tools",
            "reflect_check": "reflect_check"
        }
    )

    # tools执行完 → 同步结果 → 返回agent_node
    graph.add_edge("tools", "sync_tool_result")
    graph.add_edge("sync_tool_result", "agent_node")

    # 反思路由（已改名为route_reflect）
    graph.add_conditional_edges(
        source="reflect_check",
        path=route_reflect,
        path_map={
            "agent_node": "agent_node",
            "generate_final_ans": "generate_final_ans"
        }
    )

    graph.add_edge("generate_final_ans", END)
    log_info("Agent构建", "bind‑tools流程图构建完成")
    return graph.compile()