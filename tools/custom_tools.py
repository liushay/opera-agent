from langchain_core.tools import StructuredTool
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

import config

# 初始化向量库
embedding = OllamaEmbeddings(model=config.EMBED_MODEL)
vector_store = Chroma(
    persist_directory=config.CHROMA_PERSIST_PATH,
    embedding_function=embedding
)
retriever = vector_store.as_retriever(search_kwargs={"k": config.RETRIEVE_TOP_K})

# 工具1：知识库检索工具
def search_knowledge_base(query: str) -> str:
    """
    当用户询问RAG、LangChain、文本分块、向量库相关技术问题时调用此工具
    参数query：用户的技术问题文本
    返回知识库匹配的参考文档内容
    """
    docs = retriever.invoke(query)
    res_text = "\n".join([f"文档片段：{doc.page_content}，来源：{doc.metadata}" for doc in docs])
    return res_text

# 工具2：数学计算器工具
def calculator(a: float, b: float, op: str) -> str:
    """
    仅用于数学四则运算，用户需要计算数字时调用
    参数a：第一个数字，浮点数
    参数b：第二个数字，浮点数
    参数op：运算符，仅支持 + - * /
    返回计算结果字符串
    """
    if op == "+":
        result = a + b
    elif op == "-":
        result = a - b
    elif op == "*":
        result = a * b
    elif op == "/":
        if b == 0:
            return "错误：除数不能为0"
        result = a / b
    else:
        return "不支持该运算符，仅支持 + - * /"
    return f"计算结果：{a} {op} {b} = {result}"

# 转为LangChain标准结构化工具
knowledge_tool = StructuredTool.from_function(search_knowledge_base)
calc_tool = StructuredTool.from_function(calculator)
# 工具列表，提供给模型识别
tool_list = [knowledge_tool, calc_tool]