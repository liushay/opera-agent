from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. 修正会话存储：使用官方标准内存历史对象，而非原生list
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        # 替换 [] 为标准消息存储实例
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 2. 提示词模板
prompt = PromptTemplate(
    input_variables=["chat_history", "user_input"],
    template="""
历史对话：
{chat_history}
用户新问题：{user_input}
你是Agent开发助教，连贯回答用户问题，不要遗忘上文内容。
"""
)

# 3. 初始化模型
llm = ChatOllama(model="qwen2:7b", temperature=0.3)
# LCEL管道
chain = prompt | llm
# 绑定历史记忆
chat_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="user_input",
    history_messages_key="chat_history"
)

if __name__ == "__main__":
    print("对话启动，输入 exit 退出")
    while True:
        user_input = input("用户：")
        if user_input == "exit":
            print("对话结束，完整历史：")
            # .messages 获取结构化对话列表
            print(store["session_001"].messages)
            break
        res = chat_with_history.invoke(
            {"user_input": user_input},
            config={"configurable": {"session_id": "session_001"}}
        )
        print(f"AI：{res.content}\n")