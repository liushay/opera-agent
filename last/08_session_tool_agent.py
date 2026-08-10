from typing import TypedDict, Annotated, Sequence, Literal
import operator
import json
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from tools.custom_tools import tool_list, knowledge_tool, calc_tool

# 会话全局存储（多用户隔离）
session_store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# 状态定义
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    tool_call: dict
    tool_result: str
    reflect_times: int

llm = ChatOllama(model="qwen2:7b", temperature=0.1)
tools_info = "\n".join([f"工具名称：{t.name}，功能：{t.description}" for t in tool_list])

# 节点1：工具规划
def plan_tool_call(state: AgentState) -> AgentState:
    prompt = f"""
可用工具：
{tools_info}
1. 技术问题调用search_knowledge_base；2. 计算调用calculator；3. 闲聊none
仅输出JSON：{{"tool_name":"xxx","params":{{}}}}
问题：{state['user_query']}
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    call_data = json.loads(resp.content.strip())
    return {"tool_call": call_data}

# 节点2：执行工具
def run_tool(state: AgentState) -> AgentState:
    call_info = state["tool_call"]
    tn = call_info["tool_name"]
    params = call_info["params"]
    res = "无需工具"
    if tn == "search_knowledge_base":
        res = knowledge_tool.invoke(params)
    elif tn == "calculator":
        res = calc_tool.invoke(params)
    return {"tool_result": res}

# 节点3：反思校验
def reflect_check(state: AgentState) -> AgentState:
    q = state["user_query"]
    tr = state["tool_result"]
    cnt = state["reflect_times"]
    if cnt >= 2:
        return {"reflect_times": cnt}
    prompt = f"""
问题：{q}
现有资料：{tr}
只输出True/False，False代表需要再次检索
"""
    ans = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    if ans == "False":
        return {"reflect_times": cnt + 1}
    return {"reflect_times": cnt}

# 节点4：生成回答
def generate_final_ans(state: AgentState) -> AgentState:
    history = state["messages"]
    q = state["user_query"]
    tr = state["tool_result"]
    prompt = f"""
历史对话：{history}
用户提问：{q}
工具查询结果：{tr}
严格根据资料回答，不编造内容
"""
    reply = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=reply.content)]}

# 路由：工具分流
def route_tool(state: AgentState) -> Literal["run_tool", "generate_final_ans"]:
    if state["tool_call"]["tool_name"] == "none":
        return "generate_final_ans"
    return "run_tool"

# 路由：反思循环分流
def route_reflect(state: AgentState) -> Literal["plan_tool_call", "generate_final_ans"]:
    q = state["user_query"]
    tr = state["tool_result"]
    cnt = state["reflect_times"]
    if cnt >= 2:
        return "generate_final_ans"
    flag = llm.invoke([HumanMessage(content=f"问题：{q} 资料：{tr}，是否足够？只输出True/False")]).content.strip()
    return "plan_tool_call" if flag == "False" else "generate_final_ans"

# 构建图
def build_session_agent():
    graph = StateGraph(AgentState)
    graph.add_node("plan_tool_call", plan_tool_call)
    graph.add_node("run_tool", run_tool)
    graph.add_node("reflect_check", reflect_check)
    graph.add_node("generate_final_ans", generate_final_ans)

    graph.set_entry_point("plan_tool_call")
    graph.add_conditional_edges("plan_tool_call", route_tool, {"run_tool":"run_tool","generate_final_ans":"generate_final_ans"})
    graph.add_edge("run_tool", "reflect_check")
    graph.add_conditional_edges("reflect_check", route_reflect, {"plan_tool_call":"plan_tool_call","generate_final_ans":"generate_final_ans"})
    graph.add_edge("generate_final_ans", END)
    return graph.compile()

if __name__ == "__main__":
    agent = build_session_agent()
    # 两个不同会话，模拟多用户
    session_a = "user_001"
    session_b = "user_002"
    history_a = get_session_history(session_a)
    print(f"当前会话ID：{session_a}，输入exit切换会话/退出")
    current_sid = session_a

    while True:
        user_input = input("用户：")
        if user_input == "exit":
            print("对话结束，当前会话记录：")
            print(get_session_history(current_sid).messages)
            break
        # 切换会话示例：输入switch切换user001/user002
        if user_input == "switch":
            current_sid = session_b if current_sid == session_a else session_a
            print(f"已切换至会话：{current_sid}")
            continue
        history = get_session_history(current_sid)
        history.add_message(HumanMessage(content=user_input))
        init_state = {
            "messages": history.messages,
            "user_query": user_input,
            "tool_call": {},
            "tool_result": "",
            "reflect_times": 0
        }
        res = agent.invoke(init_state)
        ai_msg = res["messages"][-1]
        history.add_message(ai_msg)
        print(f"AI：{ai_msg.content}\n")