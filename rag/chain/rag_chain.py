from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import config
from rag.vectorstore import hybrid_search
from utils.logger import log_info
from utils.rag_exceptions import LLMModelException
from rag.vectorstore import hybrid_retrieve

class RAGChain:
    def __init__(self):
        self.llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMP)
        self.stream_llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMP, streaming=True)

    def run(self, user_query: str, history_text: str = "") -> str:
        docs = hybrid_retrieve(user_query)
        ref_content = "\n".join([d.page_content for d in docs])
        prompt = f"""
历史对话上下文：{history_text}
用户问题：{user_query}
知识库参考资料：{ref_content}
依据资料回答，禁止编造内容。
        """
        log_info("RAG链路", f"执行同步问答，检索文档{len(docs)}条")
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            return resp.content
        except Exception as e:
            raise LLMModelException("LLM调用失败", e)

    async def stream_run(self, user_query: str, history_text: str = ""):
        docs = hybrid_search(user_query)
        ref_content = "\n".join([d.page_content for d in docs])
        prompt = f"""
历史对话上下文：{history_text}
用户问题：{user_query}
知识库参考资料：{ref_content}
依据资料回答，禁止编造内容。
        """
        async for chunk in self.stream_llm.astream([HumanMessage(content=prompt)]):
            yield chunk.content

rag_chain = RAGChain()