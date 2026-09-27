# tools/custom_tools.py 单Agent可用工具注册表
# 当前工具：
#   1. search_knowledge_base 知识库混合检索
#
# 说明：计算器工具已移除（戏曲科普平台聚焦于知识检索与内容生成）
# 戏曲科普五功能（戏词解剖/人物对谈/闯关/学戏路线/脸谱）已下沉到 opera/ 业务模块，
# 由多Agent协作编排调用，不注册为单Agent的StructuredTool。
from langchain_core.tools import StructuredTool
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from rag.vectorstore import kb, bm25_kb, hybrid_retrieve
import config
from utils.logger import log_info, log_warn, log_error
from utils.rag_exceptions import VectorStoreException, BM25IndexException

embedding = OllamaEmbeddings(model=config.EMBED_MODEL)
vector_store = Chroma(
    persist_directory=config.CHROMA_PERSIST_PATH,
    embedding_function=embedding
)
retriever = vector_store.as_retriever(search_kwargs={"k": config.RETRIEVE_TOP_K})

# 工具1：知识库检索工具
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

knowledge_tool = StructuredTool.from_function(search_knowledge_base)
tool_list = [knowledge_tool]