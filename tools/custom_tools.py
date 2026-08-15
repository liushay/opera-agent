from langchain_core.tools import StructuredTool
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from rag.vectorstore import kb, bm25_kb, hybrid_retrieve
import config
from utils.logger import log_info, log_warn, log_error
from utils.rag_exceptions import VectorStoreException, BM25IndexException
from langchain_core.tools import tool

embedding = OllamaEmbeddings(model=config.EMBED_MODEL)
vector_store = Chroma(
    persist_directory=config.CHROMA_PERSIST_PATH,
    embedding_function=embedding
)
retriever = vector_store.as_retriever(search_kwargs={"k": config.RETRIEVE_TOP_K})

# 工具1：知识库检索工具
@tool
def search_knowledge_base(query: str) -> str:
    """从本地知识库检索和问题相关的文档内容
    query: 用户待检索的查询文本
    """
    log_info("知识库工具", f"执行检索工具，query={query}")
    try:
        docs = hybrid_retrieve(query)
    except (VectorStoreException, BM25IndexException) as e:
        log_error("知识库工具检索失败", "检索底层异常", e)
        # 检索异常直接向上抛出，上层Agent捕获为流程异常
        raise e

    if not docs:
        log_warn("知识库工具", "检索结果为空文档")
        return "知识库未查询到相关内容"

    res_text = "\n".join([f"文档片段：{doc.page_content}，来源：{doc.metadata}" for doc in docs])
    log_info("知识库工具", f"检索成功，返回{len(docs)}条文档")
    return res_text

# 工具2：数学计算器工具
@tool
def calculator(a: float, b: float, op: str) -> str:
    """执行两个数字的四则运算
    a: 第一个运算数字
    b: 第二个运算数字
    op: 运算符，仅支持 + - * /
    """
    log_info("计算工具", f"执行计算 {a} {op} {b}")
    if op == "+":
        result = a + b
    elif op == "-":
        result = a - b
    elif op == "*":
        result = a * b
    elif op == "/":
        if b == 0:
            log_warn("计算工具", "除数为0，计算失败")
            return "错误：除数不能为0"
        result = a / b
    else:
        log_warn("计算工具", f"不支持运算符：{op}")
        return "不支持该运算符，仅支持 + - * /"
    log_info("计算工具", f"计算完成，结果={result}")
    return f"计算结果：{a} {op} {b} = {result}"

tool_list = [search_knowledge_base, calculator]