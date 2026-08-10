from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

# 会话存储
session_store = {}
def get_session(sid: str) -> BaseChatMessageHistory:
    if sid not in session_store:
        session_store[sid] = InMemoryChatMessageHistory()
    return session_store[sid]

# 状态定义不变
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    need_retrieval: bool
    context: str

llm = ChatOllama(model="qwen2:7b", temperature=0.2)
embedding = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 节点函数和分支逻辑和上一个文件完全一致，此处省略 judge_intent / retrieve_docs / generate_answer / route_retrieval
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

# 构建图函数不变
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("judge_intent", judge_intent)
    graph.add_node("retrieve_docs", retrieve_docs)
    graph.add_node("generate_answer", generate_answer)
    graph.set_entry_point("judge_intent")
    graph.add_conditional_edges("judge_intent", route_retrieval, {"retrieve_docs":"retrieve_docs","generate_answer":"generate_answer"})
    graph.add_edge("retrieve_docs", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()

if __name__ == "__main__":
    agent = build_graph()
    session_id = "user_001"
    history = get_session(session_id)
    print("多会话隔离Agent，输入exit退出")
    while True:
        user_q = input("用户：")
        if user_q == "exit":
            print("完整会话记录：", history.messages)
            break
        history.add_message(HumanMessage(content=user_q))
        init_state = {
            "messages": history.messages,
            "user_query": user_q,
            "need_retrieval": False,
            "context": ""
        }
        res = agent.invoke(init_state)
        ai_msg = res["messages"][-1]
        history.add_message(ai_msg)
        print(f"AI：{ai_msg.content}\n")