from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2:7b", temperature=0.3)

if __name__ == "__main__":
    # 系统人设 + 用户提问
    messages = [
        SystemMessage(content="你是AI智能体教学老师，只用大白话讲解，不使用专业术语"),
        HumanMessage(content="LangGraph和LangChain有什么区别？")
    ]
    resp = llm.invoke(messages)
    print(resp.content)
    # 追加历史AI回答，实现多轮上下文
    messages.append(AIMessage(content=resp.content))
    messages.append(HumanMessage(content="那开发项目优先选哪个？"))
    resp2 = llm.invoke(messages)
    print("\n第二轮回答：")
    print(resp2.content)