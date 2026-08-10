from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END

# 1. 全局状态定义，所有节点共享该数据
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]  # 对话历史
    user_query: str  # 用户原始问题
    need_retrieval: bool  # 标记是否需要检索知识库
    context: str  # 检索到的知识库内容

# 2. 初始化模型、向量库
llm = ChatOllama(model="qwen2:7b", temperature=0.2)
embedding = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ---------------------- 定义各个节点函数 ----------------------
# 节点1：意图判断，输出是否需要检索
def judge_intent(state: AgentState) -> AgentState:
    user_q = state["user_query"]
    judge_prompt = f"""
你是意图分类器，只输出True或False。
规则：
1. 如果问题和RAG、文档分块、向量库、LangChain开发相关 → True（需要检索知识库）
2. 日常闲聊、问候、无关问题 → False（不需要检索）
用户问题：{user_q}
仅返回True/False，不要多余文字
"""
    res = llm.invoke([HumanMessage(content=judge_prompt)])
    flag = res.content.strip() == "True"
    return {"need_retrieval": flag}

# 节点2：知识库检索节点
def retrieve_docs(state: AgentState) -> AgentState:
    docs = retriever.invoke(state["user_query"])
    context_text = "\n\n".join([d.page_content for d in docs])
    return {"context": context_text}

# 节点3：生成最终回答，携带历史对话+检索内容
def generate_answer(state: AgentState) -> AgentState:
    msg_list = state["messages"]
    user_q = state["user_query"]
    ctx = state["context"]
    need_search = state["need_retrieval"]

    full_prompt = ""
    if need_search:
        full_prompt = f"""
历史对话：{msg_list}
参考知识库内容：{ctx}
用户问题：{user_q}
严格根据知识库内容回答，禁止编造信息
"""
    else:
        full_prompt = f"""
历史对话：{msg_list}
用户问题：{user_q}
友好简洁闲聊回答，无需参考知识库
"""
    answer = llm.invoke([HumanMessage(content=full_prompt)])
    # 追加AI回复到对话历史
    new_msg = AIMessage(content=answer.content)
    return {"messages": [new_msg]}

# ---------------------- 分支判断函数 ----------------------
def route_retrieval(state: AgentState):
    # 根据need_retrieval决定走哪个节点
    if state["need_retrieval"]:
        return "retrieve_docs"
    else:
        return "generate_answer"

# ---------------------- 构建图流程 ----------------------
def build_graph():
    graph = StateGraph(AgentState)
    # 注册所有节点
    graph.add_node("judge_intent", judge_intent)
    graph.add_node("retrieve_docs", retrieve_docs)
    graph.add_node("generate_answer", generate_answer)

    # 入口：从意图判断开始
    graph.set_entry_point("judge_intent")
    # 条件分支：意图判断完成后分流
    graph.add_conditional_edges(
        source="judge_intent",
        path=route_retrieval,
        path_map={
            "retrieve_docs": "retrieve_docs",
            "generate_answer": "generate_answer"
        }
    )
    # 检索完成后统一进入生成回答节点
    graph.add_edge("retrieve_docs", "generate_answer")
    # 生成回答后流程结束
    graph.add_edge("generate_answer", END)

    return graph.compile()

# 主运行逻辑
if __name__ == "__main__":
    agent = build_graph()
    print("LangGraph智能体启动，输入exit退出")
    chat_history = []
    while True:
        user_input = input("用户：")
        if user_input == "exit":
            print("对话结束")
            break
        chat_history.append(HumanMessage(content=user_input))
        init_state = {
            "messages": chat_history,
            "user_query": user_input,
            "need_retrieval": False,
            "context": ""
        }
        result = agent.invoke(init_state)
        ai_resp = result["messages"][-1].content
        print(f"AI：{ai_resp}\n")
        chat_history = result["messages"]