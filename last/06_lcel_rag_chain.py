from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 初始化模型与向量库
llm = ChatOllama(model="qwen2:7b", temperature=0.2)
embedding = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 2. RAG专用提示词
rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
依据下方参考文档回答用户问题，禁止编造信息：
参考文档：
{context}

用户问题：{question}
回答简洁准确。
"""
)

# 3. 定义文档拼接工具：把检索到的多段文本合并
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# 4. LCEL管道串联完整RAG链
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    # 单轮测试
    res = rag_chain.invoke("RAG完整开发流程是什么？")
    print(res)