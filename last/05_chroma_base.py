from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
# 1. 初始化本地嵌入模型（本地ollama启动nomic-embed-text）
embedding = OllamaEmbeddings(model="nomic-embed-text")
# 2. 初始化持久化向量库，数据保存到./chroma_db文件夹
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding
)

if __name__ == "__main__":
    # 测试文本片段
    texts = [
        "RAG检索增强生成可以解决大模型知识滞后、幻觉问题",
        "RecursiveCharacterTextSplitter分层切割文本，保证语义完整",
        "Chroma是本地轻量向量库，无需部署服务，适合本地开发调试",
        "LangChain通过Document对象统一管理各类文档元数据"
    ]
    # 存入向量库
    vector_store.add_texts(texts)
    print("文本已存入向量库")

    # 相似度检索，top_k=2 返回最相近2条结果
    query = "什么工具能解决大模型幻觉？"
    search_result = vector_store.similarity_search(query, k=2)
    print(f"\n检索问题：{query}")
    for idx, doc in enumerate(search_result):
        print(f"匹配内容{idx+1}：{doc.page_content}")

    # 清空向量库（测试用，项目正式环境慎用）
    # vector_store.delete_collection()