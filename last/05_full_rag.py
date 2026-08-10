from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from utils.doc_split_utils import DocProcessor  # 复用Day4封装的文档处理工具

# 1. 初始化模型、向量库
llm = ChatOllama(model="qwen2:7b", temperature=0.2)
embedding = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

# 2. 文档加载+分割（复用Day4工具）
processor = DocProcessor(chunk_size=120, chunk_overlap=15)
# 读取测试文档，可替换test.md/test.txt
raw_docs = processor.load_file("test.txt")
chunk_data = processor.split_docs(raw_docs)
# 提取纯文本块用于入库
chunk_texts = [item["text"] for item in chunk_data]
chunk_meta = [item["source"] for item in chunk_data]

# 3. 文本块批量存入向量库
vector_store.add_texts(texts=chunk_texts, metadatas=chunk_meta)
print("文档分块全部存入Chroma向量库")

# 4. 创建检索器，设置top_k=3
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 5. 构建RAG专用提示词模板，约束模型只根据检索内容回答
rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
请严格根据下面参考上下文回答用户问题，禁止编造不存在信息：
参考上下文：
{context}

用户问题：{question}
回答简洁清晰，不输出无关内容：
"""
)

# 6. LCEL 组装RAG链
def rag_chain_invoke(question: str):
    # 1. 检索相关文档块
    related_docs = retriever.invoke(question)
    # 拼接检索到的上下文
    context_text = "\n".join([doc.page_content for doc in related_docs])
    # 填充提示词调用大模型
    prompt_input = rag_prompt.format(context=context_text, question=question)
    res = llm.invoke(prompt_input)
    return res.content, related_docs

if __name__ == "__main__":
    while True:
        user_q = input("\n请输入你的问题（输入exit退出）：")
        if user_q == "exit":
            print("RAG对话结束")
            break
        answer, docs = rag_chain_invoke(user_q)
        print("===检索到的参考文档===")
        for d in docs:
            print(f"- {d.page_content}")
        print("\n===AI回答===")
        print(answer)