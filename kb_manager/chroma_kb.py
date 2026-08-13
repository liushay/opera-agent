from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from utils.doc_split_utils import DocProcessor
import config
from kb_manager.bm25_retriever import bm25_kb


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
        # 关键1：None兜底为空字典
        metadatas = [item["source"] or {} for item in chunk_data]

        exist_metas = self.get_all_metadata()
        new_texts, new_metas = [], []
        for t, m in zip(texts, metadatas):
            if m not in exist_metas:
                new_texts.append(t)
                new_metas.append(m)

        if len(new_texts) == 0:
            print(f"文件{file_path}内容已存在向量库，无需新增")
            return
        # 关键2：再次兜底，杜绝None
        safe_metas = [meta or {} for meta in new_metas]
        self.vector_store.add_texts(texts=new_texts, metadatas=safe_metas)
        # 同步增量更新BM25索引
        bm25_kb.add_texts(new_texts, safe_metas)
        print(f"成功新增{len(new_texts)}条文本块至向量库 & BM25索引")

    # 全量重建BM25方法
    def rebuild_full_bm25(self):
        """读取Chroma全部数据，完整重建BM25索引（初始化/清空库后调用）"""
        all_data = self.vector_store.get()
        all_texts = all_data["documents"]
        all_metas = all_data["metadatas"]
        corpus = list(zip(all_texts, all_metas))
        bm25_kb.rebuild_index(corpus)
        print("BM25索引全量重建完成")

    # 基础相似度检索 + 距离阈值过滤
    def similarity_search_filter(self, query: str):
        docs_with_score = self.vector_store.similarity_search_with_score(
            query, k=config.RETRIEVE_TOP_K
        )
        filter_docs = []
        for doc, score in docs_with_score:
            if score < config.SIMILARITY_THRESHOLD:
                # 容错处理：metadata为None时自动赋值为空字典{}，避免Pydantic校验报错
                doc.metadata = doc.metadata or {}
                filter_docs.append(doc)
        return filter_docs

    # 升级MMR检索，采用as_retriever，支持候选池fetch_k
    # 注意：score_threshold仅适用于similarity_score_threshold检索类型，
    # mmr模式下透传给max_marginal_relevance_search会报TypeError，因此移除
    def mmr_search(self, query: str):
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": config.RETRIEVE_TOP_K,
                "fetch_k": 6
            }
        )
        docs = retriever.invoke(query)
        # 容错处理：metadata为None时自动赋值为空字典{}，避免Pydantic校验报错
        for doc in docs:
            doc.metadata = doc.metadata or {}
        return docs

    # 获取库内全部元数据，用于文件增量入库去重
    def get_all_metadata(self):
        data = self.vector_store.get()
        return data["metadatas"]

    # 清空向量集合
    def clear_kb(self):
        self.vector_store.delete_collection()
        print("向量库已清空")
        self.rebuild_full_bm25()


# 全局单例，项目各处统一导入
kb = ChromaKnowledgeBase()