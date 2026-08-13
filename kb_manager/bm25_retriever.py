from rank_bm25 import BM25Okapi
from typing import List, Tuple
from langchain_core.documents import Document
import config
from utils.doc_split_utils import DocProcessor
import jieba

class BM25Retriever:
    def __init__(self):
        self.processor = DocProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        # 存储原始文本、元数据、分词文本
        self.corpus_texts: List[str] = []
        self.corpus_metas: List[dict] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25_index: BM25Okapi | None = None

    def _tokenize(self, text: str) -> List[str]:
        return jieba.lcut(text)

    def rebuild_index(self, all_docs: List[Tuple[str, dict]]):
        """全量重建BM25索引：传入(文本,元数据)列表"""
        self.corpus_texts.clear()
        self.corpus_metas.clear()
        self.tokenized_corpus.clear()
        for text, meta in all_docs:
            self.corpus_texts.append(text)
            self.corpus_metas.append(meta)
            tokens = self._tokenize(text)
            self.tokenized_corpus.append(tokens)
        if self.tokenized_corpus:
            self.bm25_index = BM25Okapi(self.tokenized_corpus)

    def add_texts(self, texts: List[str], metadatas: List[dict]):
        """增量添加文档，更新BM25索引"""
        for text, meta in zip(texts, metadatas):
            self.corpus_texts.append(text)
            self.corpus_metas.append(meta)
            tokens = self._tokenize(text)
            self.tokenized_corpus.append(tokens)
        self.bm25_index = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = config.BM25_TOP_K) -> List[Tuple[Document, float]]:
        """
        BM25检索，返回 (Document, bm25分数)
        """
        if not self.bm25_index:
            return []
        query_tokens = self._tokenize(query)
        scores = self.bm25_index.get_scores(query_tokens)
        # 绑定文档+分数并排序
        doc_score_pairs = list(zip(self.corpus_texts, self.corpus_metas, scores))
        doc_score_pairs.sort(key=lambda x: x[2], reverse=True)
        # 截取top_k
        top_pairs = doc_score_pairs[:top_k]
        res = []
        for text, meta, score in top_pairs:
            # 容错处理：metadata为None时自动赋值为空字典{}，避免Pydantic校验报错
            doc = Document(page_content=text, metadata=meta or {})
            res.append((doc, float(score)))
        return res

# 全局单例
bm25_kb = BM25Retriever()