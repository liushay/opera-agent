from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from utils.doc_split_utils import DocProcessor
import config


class ChromaKnowledgeBase:
    def __init__(self):
        self.embedding = OllamaEmbeddings(model=config.EMBED_MODEL)
        self.vector_store = Chroma(
            persist_directory=config.CHROMA_PERSIST_PATH,
            embedding_function=self.embedding
        )
        self.processor = DocProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )

    # 增量加载单个文件，依据文件source元数据去重
    def add_file_increment(self, file_path: str):
        raw_docs = self.processor.load_file(file_path)
        chunk_data = self.processor.split_docs(raw_docs)
        texts = [item["text"] for item in chunk_data]
        metadatas = [item["source"] for item in chunk_data]

        exist_metas = self.get_all_metadata()
        new_texts, new_metas = [], []
        for t, m in zip(texts, metadatas):
            if m not in exist_metas:
                new_texts.append(t)
                new_metas.append(m)

        if len(new_texts) == 0:
            print(f"文件{file_path}内容已存在向量库，无需新增")
            return
        self.vector_store.add_texts(texts=new_texts, metadatas=new_metas)
        print(f"成功新增{len(new_texts)}条文本块至向量库")

    # 基础相似度检索 + 距离阈值过滤
    def similarity_search_filter(self, query: str):
        docs_with_score = self.vector_store.similarity_search_with_score(
            query, k=config.RETRIEVE_TOP_K
        )
        filter_docs = []
        for doc, score in docs_with_score:
            if score < config.SIMILARITY_THRESHOLD:
                filter_docs.append(doc)
        return filter_docs

    # 升级MMR检索，采用as_retriever，支持相似度门槛、候选池fetch_k
    def mmr_search(self, query: str):
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": config.RETRIEVE_TOP_K,
                "fetch_k": 6,
                "score_threshold": config.SIMILARITY_THRESHOLD
            }
        )
        return retriever.invoke(query)

    # 获取库内全部元数据，用于文件增量入库去重
    def get_all_metadata(self):
        data = self.vector_store.get()
        return data["metadatas"]

    # 清空向量集合
    def clear_kb(self):
        self.vector_store.delete_collection()
        print("向量库已清空")


# 全局单例，项目各处统一导入
kb = ChromaKnowledgeBase()