from typing import List, Dict, Tuple
from langchain_core.documents import Document
import config
from kb_manager.chroma_kb import kb
from kb_manager.bm25_retriever import bm25_kb
from utils.logger import print_log
from utils.cache_utils import get_retrieve_cache, set_retrieve_cache, serialize_docs, deserialize_docs

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
    cache_raw = get_retrieve_cache(query)
    if cache_raw is not None:
        cached_docs = deserialize_docs(cache_raw)
        print_log("混合检索", f"命中检索缓存，直接返回{len(cached_docs)}条文档")
        return cached_docs
    """
    混合检索统一入口：BM25关键词 + Chroma向量融合重排
    关闭混合检索时自动降级为原有MMR向量检索
    """
    if not config.ENABLE_HYBRID_SEARCH:
        vec_docs = kb.mmr_search(query)
        # 写入缓存
        set_retrieve_cache(query, serialize_docs(vec_docs))
        return vec_docs

    # 1. 两路召回
    vec_docs = kb.mmr_search(query)  # 向量召回
    bm25_pairs = bm25_kb.search(query, top_k=config.BM25_TOP_K)  # (doc, bm25分数)

    # 2. 构建文档-分数映射
    doc_map: Dict[str, Tuple[Document, float, float]] = {}  # key:唯一标识, (doc, vec_norm, bm25_norm)
    vec_raw_scores = []
    # 向量检索无原生分数，统一赋值1递减模拟相似度分（MMR靠前=高分）
    for idx, doc in enumerate(vec_docs):
        unique_key = f"{doc.page_content}_{doc.metadata}"
        vec_raw_scores.append(1 - idx / len(vec_docs))
        doc_map[unique_key] = [doc, 1 - idx / len(vec_docs), 0.0]

    # BM25原始分数收集
    bm25_raw_scores = [p[1] for p in bm25_pairs]
    norm_bm25 = _normalize_score(bm25_raw_scores)
    for idx, (doc, raw_score) in enumerate(bm25_pairs):
        unique_key = f"{doc.page_content}_{doc.metadata}"
        bm25_norm = norm_bm25[idx]
        if unique_key in doc_map:
            # 两路同时命中，更新bm25归一分
            doc_map[unique_key][2] = bm25_norm
        else:
            # 仅BM25命中，向量分0
            doc_map[unique_key] = [doc, 0.0, bm25_norm]

    # 3. 加权融合计算总分
    merge_list = []
    vec_w = config.VECTOR_WEIGHT
    bm25_w = config.BM25_WEIGHT
    for data in doc_map.values():
        doc, vec_norm, bm25_norm = data
        total = vec_norm * vec_w + bm25_norm * bm25_w
        merge_list.append((doc, total))

    # 4. 降序排序、去重、截断最终条数
    merge_list.sort(key=lambda x: x[1], reverse=True)
    final_docs = [item[0] for item in merge_list[:config.HYBRID_FINAL_K]]
    # 在当前函数内执行日志输出
    print_log(tag="混合检索",
              content=f"向量召回{len(vec_docs)}条，BM25召回{len(bm25_pairs)}条，融合后返回{len(final_docs)}条")
    # 将融合后的文档写入缓存
    set_retrieve_cache(query, serialize_docs(final_docs))
    return final_docs