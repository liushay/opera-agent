import time
import httpx
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from rag.loader import document_loader
from rag.splitter import text_splitter
import config
from utils.logger import log_info, log_error, log_warn
from utils.rag_exceptions import VectorStoreException

# MMR检索Ollama连接失败重试配置
MMR_RETRY_MAX = 3          # 最大重试次数
MMR_RETRY_BASE_DELAY = 2.0  # 基础退避延迟（秒）

class ChromaKnowledgeBase:
    def __init__(self):
        try:
            self.embedding = OllamaEmbeddings(model=config.EMBED_MODEL)
            self.vector_store = Chroma(
                persist_directory=config.CHROMA_PERSIST_PATH,
                embedding_function=self.embedding
            )
            log_info("向量库初始化", "Chroma向量库加载完成")
        except Exception as e:
            err = VectorStoreException("Chroma向量库初始化失败，磁盘路径无权限或嵌入模型不可用", e)
            log_error("向量库初始化失败", err.msg, e)
            raise err

    # 增量加载单个文件，依据文件source元数据去重
    def add_file_increment(self, file_path: str):
        from rag.vectorstore import bm25_kb
        raw_docs = document_loader.load(file_path)
        chunk_data = text_splitter.split_documents(raw_docs)
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
        # 1、先写入Chroma向量库
        self.vector_store.add_texts(texts=new_texts, metadatas=safe_metas)
        # 2、紧接着同步增量更新BM25索引
        bm25_kb.add_texts(new_texts, safe_metas)

        print(f"成功新增{len(new_texts)}条文本块至向量库 & BM25索引")

    # 全量重建BM25方法
    def rebuild_full_bm25(self):
        """读取Chroma全部数据，完整重建BM25索引（初始化/清空库后调用）"""
        from rag.vectorstore import bm25_kb
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

    # MMR检索：直接调用max_marginal_relevance_search，避免as_retriever包装层兼容性问题
    def mmr_search(self, query: str):
        """带重试的MMR检索：Ollama嵌入调用临时断开时自动重试+指数退避，避免单次连接波动导致整个检索崩溃"""
        last_err = None
        for attempt in range(1, MMR_RETRY_MAX + 1):
            try:
                # fetch_k 使用 VECTOR_TOP_K 确保候选池足够大（≥k），同时用 lambda_mult=0.5 平衡相关性与多样性
                docs = self.vector_store.max_marginal_relevance_search(
                    query,
                    k=config.RETRIEVE_TOP_K,
                    fetch_k=max(config.VECTOR_TOP_K, config.RETRIEVE_TOP_K * 2),
                    lambda_mult=0.5,
                )
                # 容错处理：metadata为None时自动赋值为空字典{}，避免Pydantic校验报错
                for doc in docs:
                    doc.metadata = doc.metadata or {}
                if attempt > 1:
                    log_info("向量库MMR检索", f"第{attempt}次重试成功")
                return docs
            except (httpx.ConnectError, httpx.RemoteProtocolError, ConnectionError) as e:
                last_err = e
                if attempt < MMR_RETRY_MAX:
                    delay = MMR_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    log_warn("向量库MMR检索", f"第{attempt}次Ollama连接失败，{delay:.1f}s后重试：{e}")
                    time.sleep(delay)
                else:
                    log_error("向量库MMR检索", f"重试{MMR_RETRY_MAX}次后仍失败", e)
            except Exception as e:
                # 非连接类异常不重试，直接抛出
                raise VectorStoreException(f"向量库MMR检索执行失败: {e}", e)
        raise VectorStoreException(f"向量库MMR检索失败：Ollama连接{MMR_RETRY_MAX}次重试后仍不可用", last_err)

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