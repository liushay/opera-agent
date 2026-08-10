from typing import TypedDict, Annotated, Sequence, Literal
import operator
import json
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from tools.custom_tools import tool_list, knowledge_tool, calc_tool

# 全局状态定义
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    tool_call: dict  # 存储模型生成的工具调用参数
    tool_result: str  # 工具执行返回结果

llm = ChatOllama(model="qwen2:7b", temperature=0.2)
tools_info = "\n".join([f"工具名称：{t.name}，功能：{t.description}" for t in tool_list])

# 节点1：模型判断是否调用工具，输出调用参数
def plan_tool_call(state: AgentState) -> AgentState:
    prompt = f"""
你拥有以下可用工具：
{tools_info}
规则：
1. 用户问RAG、分块、向量库等技术问题 → 调用search_knowledge_base，传入query参数
2. 用户需要数字四则计算 → 调用calculator，传入a、b、op三个参数
3. 普通闲聊、无计算无技术查询 → 不调用任何工具
输出要求：
需要调用工具输出JSON格式：{{"tool_name":"工具名","params":{{参数键值对}}}}
不需要调用工具输出：{{"tool_name":"none"}}
用户问题：{state['user_query']}
仅输出JSON，无多余文字
"""
    msg = HumanMessage(content=prompt)
    resp = llm.invoke([msg])
    call_data = json.loads(resp.content.strip())
    return {"tool_call": call_data}

# 节点2：执行对应工具
def run_tool(state: AgentState) -> AgentState:
    call_info = state["tool_call"]
    tool_name = call_info["tool_name"]
    params = call_info["params"]
    output = ""
    if tool_name == "search_knowledge_base":
        output = knowledge_tool.invoke(params)
    elif tool_name == "calculator":
        output = calc_tool.invoke(params)
    else:
        output = "无需调用工具"
    return {"tool_result": output}

# 节点3：生成最终回答
def generate_final_ans(state: AgentState) -> AgentState:
    history = state["messages"]
    q = state["user_query"]
    tool_res = state["tool_result"]
    full_prompt = f"""
历史对话：{history}
用户问题：{q}
工具执行结果：{tool_res}
根据工具结果/历史对话完整回答，禁止编造信息
"""
    ans = llm.invoke([HumanMessage(content=full_prompt)])
    return {"messages": [AIMessage(content=ans.content)]}

# 条件路由：判断是否需要执行工具
def route_by_tool(state: AgentState) -> Literal["run_tool", "generate_final_ans"]:
    if state["tool_call"]["tool_name"] == "none":
        return "generate_final_ans"
    else:
        return "run_tool"

# 搭建图流程
def build_multi_tool_graph():
    graph = StateGraph(AgentState)
    graph.add_node("plan_tool_call", plan_tool_call)
    graph.add_node("run_tool", run_tool)
    graph.add_node("generate_final_ans", generate_final_ans)

    graph.set_entry_point("plan_tool_call")
    graph.add_conditional_edges("plan_tool_call", route_by_tool, {"run_tool":"run_tool", "generate_final_ans":"generate_final_ans"})
    graph.add_edge("run_tool", "generate_final_ans")
    graph.add_edge("generate_final_ans", END)
    return graph.compile()

if __name__ == "__main__":
    agent = build_multi_tool_graph()
    print("多工具智能体启动，输入exit退出")
    chat_history = []
    while True:
        user_input = input("用户：")
        if user_input == "exit":
            break
        chat_history.append(HumanMessage(content=user_input))
        init_state = {
            "messages": chat_history,
            "user_query": user_input,
            "tool_call": {},
            "tool_result": ""
        }
        res = agent.invoke(init_state)
        reply = res["messages"][-1].content
        print(f"AI：{reply}\n")
        chat_history = res["messages"]