from langchain_ollama import ChatOllama

# 初始化本地大模型
llm = ChatOllama(
    model="qwen2:7b",
    temperature=0.3  # 越小回答越严谨，越大创造性越强
)

if __name__ == "__main__":
    answer = llm.invoke("简单介绍什么是Agent智能体")
    print(answer.content)