from typing import List, Dict, Tuple
from langchain_core.documents import Document
import config
from kb_manager.chroma_kb import kb
from kb_manager.bm25_retriever import bm25_kb
from utils.logger import log_debug, log_info, log_warn, log_error
from utils.cache_utils import get_retrieve_cache, set_retrieve_cache, delete_retrieve_cache, serialize_docs, deserialize_docs
from utils.rag_exceptions import VectorStoreException, BM25IndexException, CacheSerializeException

def _normalize_score(scores: List[float]) -> List[float]:
    """最小-最大归一化到 [0,1]"""
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [1.0 for _ in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]

def hybrid_retrieve(query: str) -> List[Document]:
    """混合检索统一入口：BM25关键词 + Chroma向量融合重排"""
    log_info("混合检索", f"开始执行检索，query={query}")
    # 1. 读取缓存，捕获缓存脏数据异常
    try:
        cache_raw = get_retrieve_cache(query)
    except CacheSerializeException:
        log_warn("混合检索", "缓存脏数据已自动清理，重新执行检索")
        cache_raw = None

    if cache_raw is not None:
        cached_docs = deserialize_docs(cache_raw)
        if len(cached_docs) == 0:
            log_warn("混合检索", "缓存命中但文档为空，清除脏缓存")
            delete_retrieve_cache(query)
        else:
            log_info("混合检索", f"命中检索缓存，直接返回{len(cached_docs)}条文档")
            return cached_docs

    # 关闭混合检索，纯向量检索分支
    if not config.ENABLE_HYBRID_SEARCH:
        log_info("混合检索", "混合检索开关关闭，降级纯MMR向量检索")
        try:
            vec_docs = kb.mmr_search(query)
        except Exception as e:
            err_msg = "向量库MMR检索失败"
            log_error("混合检索向量库异常", err_msg, e)
            raise VectorStoreException(err_msg, e)
        set_retrieve_cache(query, serialize_docs(vec_docs))
        return vec_docs

    # 2. 两路召回，捕获向量库、BM25异常
    try:
        vec_docs = kb.mmr_search(query)
    except Exception as e:
        err_msg = "向量库MMR检索执行失败"
        log_error("混合检索向量库异常", err_msg, e)
        raise VectorStoreException(err_msg, e)

    try:
        bm25_pairs = bm25_kb.search(query, top_k=config.BM25_TOP_K)
    except BM25IndexException as e:
        log_warn("混合检索", "BM25索引异常，仅使用向量检索结果", e)
        bm25_pairs = []
    except Exception as e:
        err_msg = "BM25检索执行失败"
        log_error("混合检索BM25异常", err_msg, e)
        raise BM25IndexException(err_msg, e)

    # 3. 文档分数融合逻辑
    doc_map: Dict[str, Tuple[Document, float, float]] = {}
    vec_raw_scores = []
    for idx, doc in enumerate(vec_docs):
        unique_key = f"{doc.page_content}_{doc.metadata}"
        vec_raw_scores.append(1 - idx / len(vec_docs))
        doc_map[unique_key] = [doc, 1 - idx / len(vec_docs), 0.0]

    bm25_raw_scores = [p[1] for p in bm25_pairs]
    norm_bm25 = _normalize_score(bm25_raw_scores)
    for idx, (doc, raw_score) in enumerate(bm25_pairs):
        unique_key = f"{doc.page_content}_{doc.metadata}"
        bm25_norm = norm_bm25[idx]
        if unique_key in doc_map:
            doc_map[unique_key][2] = bm25_norm
        else:
            doc_map[unique_key] = [doc, 0.0, bm25_norm]

    vec_w = config.VECTOR_WEIGHT
    bm25_w = config.BM25_WEIGHT
    merge_list = []
    for data in doc_map.values():
        doc, vec_norm, bm25_norm = data
        total = vec_norm * vec_w + bm25_norm * bm25_w
        merge_list.append((doc, total))

    merge_list.sort(key=lambda x: x[1], reverse=True)
    final_docs = [item[0] for item in merge_list[:config.HYBRID_FINAL_K]]

    log_info("混合检索", f"向量召回{len(vec_docs)}条，BM25召回{len(bm25_pairs)}条，融合后返回{len(final_docs)}条")
    if len(final_docs) == 0:
        log_warn("混合检索", "本次检索未匹配到任何文档")

    set_retrieve_cache(query, serialize_docs(final_docs))
    return final_docs