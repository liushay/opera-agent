from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory

# 1. 模型&向量库初始化
llm = ChatOllama(model="qwen2:7b", temperature=0.2)
embedding = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 会话存储
session_store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# 提示词
memory_rag_prompt = PromptTemplate(
    input_variables=["chat_history", "context", "user_input"],
    template="""
历史对话记录：
{chat_history}

参考知识库文档：
{context}

用户当前提问：{user_input}
结合历史对话与参考文档连贯回答，不要遗忘上文，禁止编造内容。
"""
)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# --------------------------关键修复点--------------------------
# 分层定义：先做纯文本检索子链，只接收字符串query
retrieval_sub_chain = RunnablePassthrough() | retriever | format_docs

# 完整主链：单独提取user_input文本传给检索子链，避免字典流入向量检索
base_chain = (
    {
        "context": lambda x: retrieval_sub_chain.invoke(x["user_input"]),
        "user_input": lambda x: x["user_input"],
        "chat_history": lambda x: x["chat_history"]
    }
    | memory_rag_prompt
    | llm
    | StrOutputParser()
)
# -------------------------------------------------------------

# 绑定会话记忆
memory_rag_chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="user_input",
    history_messages_key="chat_history"
)

if __name__ == "__main__":
    sid = "session_001"
    print("带记忆RAG对话，输入exit结束")
    while True:
        user_q = input("用户：")
        if user_q == "exit":
            print("对话结束，完整会话记录：")
            print(session_store[sid].messages)
            break
        # 标准字典入参
        ans = memory_rag_chain.invoke(
            {"user_input": user_q},
            config={"configurable": {"session_id": sid}}
        )
        print(f"AI：{ans}\n")