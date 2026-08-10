from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

llm = ChatOllama(model="qwen2:7b", temperature=0.2)
embedding = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
根据参考文档回答问题，最后附上引用来源：
{context}
问题：{question}
回答完成后单独一行输出【引用来源】：文档路径+页码
"""
)

def format_docs_with_source(docs):
    content_list = []
    for doc in docs:
        source_info = f"【来源：{doc.metadata}】\n{doc.page_content}"
        content_list.append(source_info)
    return "\n\n".join(content_list)

rag_chain = (
    {"context": retriever | format_docs_with_source, "question": RunnablePassthrough()}
    | prompt
    | llm
)

if __name__ == "__main__":
    res = rag_chain.invoke("文本分割chunk_overlap作用是什么？")
    print(res.content)