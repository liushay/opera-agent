from typing import TypedDict, Annotated, Sequence, Literal
import operator
import json
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
import config
from tools.custom_tools import tool_list, knowledge_tool, calc_tool
from utils.logger import print_log
# 导入异常捕获装饰器
from utils.exception_handler import global_exception_handler

# 全局Agent状态定义
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    tool_call: dict
    tool_result: str
    reflect_times: int

# 初始化LLM，读取统一配置
llm = ChatOllama(
    model=config.LLM_MODEL,
    temperature=config.LLM_TEMP
)
tools_info = "\n".join([f"工具名称：{t.name}，功能：{t.description}" for t in tool_list])

# 节点1：工具规划，模型选择需要调用的工具
@global_exception_handler
def plan_tool_call(state: AgentState) -> AgentState:
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
    resp = llm.invoke([HumanMessage(content=prompt)])
    call_data = json.loads(resp.content.strip())
    return {"tool_call": call_data}

# 节点2：执行选中的工具
@global_exception_handler
def run_tool(state: AgentState) -> AgentState:
    call_info = state["tool_call"]
    tool_name = call_info["tool_name"]
    params = call_info["params"]
    res = "无需调用任何工具"

    if tool_name == "search_knowledge_base":
        res = knowledge_tool.invoke(params)
    elif tool_name == "calculator":
        res = calc_tool.invoke(params)
    return {"tool_result": res}

# 节点3：反思校验节点，判断信息是否充足
@global_exception_handler
def reflect_check(state: AgentState) -> AgentState:
    q = state["user_query"]
    tool_res = state["tool_result"]
    reflect_cnt = state["reflect_times"]
    max_reflect = config.MAX_REFLECT_TIMES

    # 达到最大反思次数，不再重复检索
    if reflect_cnt >= max_reflect:
        return {"reflect_times": reflect_cnt}

    prompt = f"""
用户问题：{q}
当前工具返回资料：{tool_res}
判断规则：仅输出 True / False
True = 现有信息足够完整回答用户问题
False = 信息缺失，需要重新检索知识库补充内容
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    flag = resp.content.strip() == "True"
    if not flag:
        return {"reflect_times": reflect_cnt + 1}
    return {"reflect_times": reflect_cnt}

# 节点4：整合对话与工具结果，生成最终回答
@global_exception_handler
def generate_final_ans(state: AgentState) -> AgentState:
    history = state["messages"]
    q = state["user_query"]
    tool_res = state["tool_result"]

    prompt = f"""
历史对话上下文：{history}
用户当前提问：{q}
工具查询参考资料：{tool_res}
要求：严格依据提供资料回答，禁止编造未出现信息，回答简洁通顺。
"""
    ans = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=ans.content)]}

# 路由1：判断是否需要执行工具
def route_by_tool(state: AgentState) -> Literal["run_tool", "generate_final_ans"]:
    if state["tool_call"]["tool_name"] == "none":
        return "generate_final_ans"
    return "run_tool"

# 路由2：反思分支，控制循环检索逻辑
def route_reflect(state: AgentState) -> Literal["plan_tool_call", "generate_final_ans"]:
    q = state["user_query"]
    tool_res = state["tool_result"]
    reflect_cnt = state["reflect_times"]
    max_reflect = config.MAX_REFLECT_TIMES

    if reflect_cnt >= max_reflect:
        return "generate_final_ans"

    judge_prompt = f"问题：{q}，现有资料：{tool_res}，信息是否充足？只输出True/False"
    res = llm.invoke([HumanMessage(content=judge_prompt)]).content.strip()
    if res == "False":
        return "plan_tool_call"
    else:
        return "generate_final_ans"

# 构建完整LangGraph智能体图（对外暴露入口）
def build_agent_graph():
    graph = StateGraph(AgentState)
    # 注册全部节点
    graph.add_node("plan_tool_call", plan_tool_call)
    graph.add_node("run_tool", run_tool)
    graph.add_node("reflect_check", reflect_check)
    graph.add_node("generate_final_ans", generate_final_ans)

    # 图入口
    graph.set_entry_point("plan_tool_call")
    # 工具判断分支
    graph.add_conditional_edges(
        source="plan_tool_call",
        path=route_by_tool,
        path_map={
            "run_tool": "run_tool",
            "generate_final_ans": "generate_final_ans"
        }
    )
    # 工具执行后进入反思节点
    graph.add_edge("run_tool", "reflect_check")
    # 反思分流：重新检索 / 直接生成答案
    graph.add_conditional_edges(
        source="reflect_check",
        path=route_reflect,
        path_map={
            "plan_tool_call": "plan_tool_call",
            "generate_final_ans": "generate_final_ans"
        }
    )
    # 生成回答后流程结束
    graph.add_edge("generate_final_ans", END)

    return graph.compile()