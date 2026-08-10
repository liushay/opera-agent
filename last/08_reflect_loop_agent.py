from typing import TypedDict, Annotated, Sequence, Literal
import operator
import json
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from tools.custom_tools import tool_list, knowledge_tool, calc_tool

# 全局状态
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    tool_call: dict
    tool_result: str
    reflect_times: int  # 限制最大反思轮次，防止死循环

llm = ChatOllama(model="qwen2:7b", temperature=0.1)
tools_info = "\n".join([f"工具名称：{t.name}，功能：{t.description}" for t in tool_list])

# 节点1：工具规划
def plan_tool_call(state: AgentState) -> AgentState:
    prompt = f"""
可用工具列表：
{tools_info}
规则：
1. RAG/分块/向量库技术问题：调用search_knowledge_base
2. 数字四则运算：调用calculator
3. 闲聊：{{"tool_name":"none"}}
输出仅JSON，无多余文字
格式：{{"tool_name":"xxx","params":{{}}}}
用户问题：{state['user_query']}
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    call_data = json.loads(resp.content.strip())
    return {"tool_call": call_data}

# 节点2：执行工具
def run_tool(state: AgentState) -> AgentState:
    call_info = state["tool_call"]
    tool_name = call_info["tool_name"]
    params = call_info["params"]
    res = "无需调用工具"
    if tool_name == "search_knowledge_base":
        res = knowledge_tool.invoke(params)
    elif tool_name == "calculator":
        res = calc_tool.invoke(params)
    return {"tool_result": res}

# 节点3：反思校验节点（新增核心）
def reflect_check(state: AgentState) -> AgentState:
    q = state["user_query"]
    tool_res = state["tool_result"]
    reflect_cnt = state["reflect_times"]
    # 最大反思2轮，避免无限循环
    if reflect_cnt >= 2:
        return {"reflect_times": reflect_cnt}

    prompt = f"""
用户问题：{q}
当前工具查询结果：{tool_res}
判断：现有信息是否足够完整回答用户问题？
仅输出 True / False
True：信息充足，可以直接回答
False：信息不足，需要再次检索知识库补充内容
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    flag = resp.content.strip() == "True"
    if not flag:
        # 信息不足，反思次数+1，回流重新检索
        return {"reflect_times": reflect_cnt + 1}
    return {"reflect_times": reflect_cnt}

# 节点4：生成最终答案
def generate_final_ans(state: AgentState) -> AgentState:
    history = state["messages"]
    q = state["user_query"]
    tool_res = state["tool_result"]
    prompt = f"""
历史对话：{history}
用户问题：{q}
工具查询内容：{tool_res}
依据现有内容完整回答，禁止编造信息
"""
    ans = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=ans.content)]}

# 路由1：判断是否执行工具
def route_by_tool(state: AgentState) -> Literal["run_tool", "generate_final_ans"]:
    if state["tool_call"]["tool_name"] == "none":
        return "generate_final_ans"
    return "run_tool"

# 路由2：反思分支（核心循环）
def route_reflect(state: AgentState) -> Literal["plan_tool_call", "generate_final_ans"]:
    q = state["user_query"]
    tool_res = state["tool_result"]
    reflect_cnt = state["reflect_times"]
    # 超过最大轮次直接回答
    if reflect_cnt >= 2:
        return "generate_final_ans"
    # 判断信息是否充足
    prompt = f"""
问题：{q}
已有资料：{tool_res}
只输出True/False，True代表信息充足，False代表需要重新检索
"""
    res = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    if res == "False":
        return "plan_tool_call"  # 回流重新调用工具检索
    else:
        return "generate_final_ans"

# 构建图
def build_reflect_agent():
    graph = StateGraph(AgentState)
    # 注册所有节点
    graph.add_node("plan_tool_call", plan_tool_call)
    graph.add_node("run_tool", run_tool)
    graph.add_node("reflect_check", reflect_check)
    graph.add_node("generate_final_ans", generate_final_ans)

    graph.set_entry_point("plan_tool_call")
    # 工具判断分支
    graph.add_conditional_edges("plan_tool_call", route_by_tool, {
        "run_tool": "run_tool",
        "generate_final_ans": "generate_final_ans"
    })
    # 工具执行完进入反思
    graph.add_edge("run_tool", "reflect_check")
    # 反思后分流：重新检索 / 生成答案
    graph.add_conditional_edges("reflect_check", route_reflect, {
        "plan_tool_call": "plan_tool_call",
        "generate_final_ans": "generate_final_ans"
    })
    graph.add_edge("generate_final_ans", END)
    return graph.compile()

if __name__ == "__main__":
    agent = build_reflect_agent()
    print("反思循环智能体启动，输入exit退出")
    chat_history = []
    while True:
        user_input = input("用户：")
        if user_input == "exit":
            break
        init_state = {
            "messages": chat_history,
            "user_query": user_input,
            "tool_call": {},
            "tool_result": "",
            "reflect_times": 0
        }
        out = agent.invoke(init_state)
        reply = out["messages"][-1].content
        print(f"AI：{reply}\n")
        chat_history = out["messages"]