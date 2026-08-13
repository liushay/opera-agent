import hashlib
import json
import utils.redis_client as redis_module
import config
from utils.logger import print_log

def _get_redis_client():
    """动态获取最新 redis_client 实例（避免 from-import 值绑定导致永远为 None）"""
    return redis_module.redis_client

def normalize_query(text: str) -> str:
    """统一格式化问题：去除换行、多余空格，用于生成缓存key"""
    if not config.CACHE_NORMALIZE_WHITESPACE:
        return text.strip()
    return " ".join(text.strip().split())

def get_query_hash(query: str) -> str:
    """将标准化后的问题转为md5哈希，缩短缓存key长度"""
    norm_q = normalize_query(query)
    return hashlib.md5(norm_q.encode("utf-8")).hexdigest()

# ========= 检索缓存通用方法 =========
def get_retrieve_cache(query: str):
    """读取检索缓存，返回序列化文档列表；无缓存返回None"""
    redis_client = _get_redis_client()
    if not config.ENABLE_RAG_CACHE or redis_client is None:
        return None
    key = f"{config.RETRIEVE_CACHE_PREFIX}{get_query_hash(query)}"
    data = redis_client.get(key)
    if data is None:
        return None
    try:
        return json.loads(data)
    except Exception as e:
        print_log("检索缓存", f"缓存数据解析失败，清除脏数据 {str(e)}")
        redis_client.delete(key)
        return None

def set_retrieve_cache(query: str, docs_data: list):
    """写入检索结果缓存，自动设置过期时间"""
    redis_client = _get_redis_client()
    if not config.ENABLE_RAG_CACHE or redis_client is None:
        return
    key = f"{config.RETRIEVE_CACHE_PREFIX}{get_query_hash(query)}"
    try:
        json_str = json.dumps(docs_data, ensure_ascii=False)
        redis_client.setex(key, config.RETRIEVE_CACHE_TTL, json_str)
        print_log("检索缓存", f"写入缓存成功 key={key}")
    except Exception as e:
        print_log("检索缓存", f"写入缓存失败 {str(e)}")

# ========= 问答会话缓存通用方法 =========
def get_chat_cache(session_id: str, query: str):
    """会话+问题双维度缓存，区分不同用户会话"""
    redis_client = _get_redis_client()
    if not config.ENABLE_RAG_CACHE or redis_client is None:
        return None
    hash_q = get_query_hash(query)
    key = f"{config.CHAT_CACHE_PREFIX}{session_id}:{hash_q}"
    data = redis_client.get(key)
    if data is None:
        return None
    try:
        return json.loads(data)["reply"]
    except Exception as e:
        print_log("问答缓存", f"缓存解析异常，删除脏key {str(e)}")
        redis_client.delete(key)
        return None

def set_chat_cache(session_id: str, query: str, reply: str):
    """保存会话问答缓存"""
    redis_client = _get_redis_client()
    if not config.ENABLE_RAG_CACHE or redis_client is None:
        return
    hash_q = get_query_hash(query)
    key = f"{config.CHAT_CACHE_PREFIX}{session_id}:{hash_q}"
    store_data = json.dumps({"query": query, "reply": reply}, ensure_ascii=False)
    redis_client.setex(key, config.CHAT_CACHE_TTL, store_data)
    print_log("问答缓存", f"会话{session_id}写入问答缓存 key={key}")

# ========= 缓存清理工具 =========
def clear_all_rag_cache():
    """清空所有RAG相关缓存"""
    redis_client = _get_redis_client()
    if redis_client is None:
        return
    retrieve_keys = redis_client.keys(f"{config.RETRIEVE_CACHE_PREFIX}*")
    chat_keys = redis_client.keys(f"{config.CHAT_CACHE_PREFIX}*")
    all_keys = retrieve_keys + chat_keys
    if all_keys:
        redis_client.delete(*all_keys)
        print_log("缓存清理", f"清空缓存共{len(all_keys)}条")
    return len(all_keys)

# ========文档序列化工具=======
def serialize_docs(docs):
    """Document列表转可序列化字典数组"""
    res = []
    for doc in docs:
        res.append({
            "page_content": doc.page_content,
            # 容错处理：metadata为None时自动赋值为空字典{}，避免Pydantic校验报错
            "metadata": doc.metadata or {}
        })
    return res

def deserialize_docs(doc_dict_list):
    """序列化字典还原Document对象"""
    from langchain_core.documents import Document
    docs = []
    for d in doc_dict_list:
        try:
            # 容错处理1：page_content键缺失或为空时跳过该条（不创建空文档），
            # 防止脏缓存产生"看似有效实际无效"的文档导致上层重试死循环
            page_content = d.get("page_content") or ""
            if not page_content:
                continue
            # 容错处理2：metadata键缺失或为None时自动赋值为空字典{}，避免Pydantic校验报错
            metadata = d.get("metadata") or {}
            docs.append(Document(page_content=page_content, metadata=metadata))
        except Exception:
            # 单条文档反序列化失败则跳过该条，防止脏缓存导致整体失败并触发上层死循环重试
            continue
    return docs

def delete_retrieve_cache(query: str):
    """删除指定查询的检索缓存（脏数据处理，防止死循环重复命中同一缓存）"""
    redis_client = _get_redis_client()
    if redis_client is None:
        return
    key = f"{config.RETRIEVE_CACHE_PREFIX}{get_query_hash(query)}"
    redis_client.delete(key)
    print_log("检索缓存", f"清除检索脏缓存 key={key}")
