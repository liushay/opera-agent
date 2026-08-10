from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# 初始化大模型
llm = ChatOllama(model="qwen2:7b", temperature=0.3)

# 1. 定义带占位符的提示词模板
prompt = PromptTemplate(
    input_variables=["question", "role"],
    template="""你是专业{role}，**仅从大模型AI领域定义作答**，禁止其他行业释义。
简洁回答用户问题：{question}，控制在100字以内。"""
)

if __name__ == "__main__":
    # 填充变量生成完整prompt文本
    input_text = prompt.format(role="Agent开发工程师", question="什么是RAG？")
    print("拼接后提示词：")
    print(input_text)
    # 传入模型调用
    res = llm.invoke(input_text)
    print("\n模型回答：")
    print(res.content)