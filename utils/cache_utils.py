import hashlib
import json
import redis
import utils.redis_client as redis_module
import config
from utils.logger import log_debug, log_info, log_warn, log_error
from utils.rag_exceptions import RedisStorageException, CacheSerializeException
from langchain_core.documents import Document

def _get_redis_client():
    """动态获取redis实例，避免静态导入None陷阱"""
    return redis_module.redis_client

def normalize_query(text: str) -> str:
    """统一格式化问题，消除空格/换行差异，保证相同语义问句命中同一缓存key"""
    if not config.CACHE_NORMALIZE_WHITESPACE:
        return text.strip()
    # 清空所有空白字符：空格、tab、换行
    return "".join(text.split())

def get_query_hash(query: str) -> str:
    """问句MD5哈希，缩短Redis key长度"""
    norm_q = normalize_query(query)
    return hashlib.md5(norm_q.encode("utf-8")).hexdigest()

# ===================== 检索缓存读写 =====================
def get_retrieve_cache(query: str):
    """
    读取向量检索缓存
    异常细分：Redis未初始化、Redis网络报错、JSON解析脏数据
    脏缓存自动删除并抛出序列化异常，上层可捕获降级
    """
    redis_client = _get_redis_client()
    cache_switch = config.ENABLE_RAG_CACHE
    key = f"{config.RETRIEVE_CACHE_PREFIX}{get_query_hash(query)}"

    # 1. 全局缓存开关关闭，直接跳过
    if not cache_switch:
        log_debug("检索缓存读取", f"缓存总开关关闭，跳过读取key={key}")
        return None

    # 2. Redis客户端未初始化，警告降级不阻断业务
    if redis_client is None:
        log_warn("检索缓存读取", f"Redis未初始化，禁用缓存读取 key={key}")
        return None

    # 3. Redis读取网络异常捕获
    try:
        raw_data = redis_client.get(key)
    except redis.RedisError as e:
        err_msg = f"Redis读取缓存失败 key={key}"
        log_error("检索缓存Redis异常", err_msg, e)
        raise RedisStorageException(err_msg, e)

    # 4. 无缓存数据，直接返回
    if raw_data is None:
        log_debug("检索缓存读取", f"未命中缓存 key={key}")
        return None

    # 5. JSON反序列化脏数据捕获
    try:
        cache_data = json.loads(raw_data)
    except Exception as e:
        err_msg = f"缓存key={key}数据非法，解析失败"
        log_error("检索缓存脏数据", err_msg, e)
        # 自动删除脏缓存，避免永久失效
        redis_client.delete(key)
        raise CacheSerializeException(err_msg, e)

    log_info("检索缓存读取", f"命中检索缓存 key={key}")
    return cache_data

def set_retrieve_cache(query: str, docs_data: list):
    """写入检索文档缓存，捕获Redis写入异常，写入失败仅日志不抛异常（不阻断主流程）"""
    redis_client = _get_redis_client()
    cache_switch = config.ENABLE_RAG_CACHE
    key = f"{config.RETRIEVE_CACHE_PREFIX}{get_query_hash(query)}"

    if not cache_switch:
        log_debug("检索缓存写入", f"缓存开关关闭，跳过写入 key={key}")
        return
    if redis_client is None:
        log_warn("检索缓存写入", f"Redis未初始化，无法写入缓存 key={key}")
        return

    try:
        json_str = json.dumps(docs_data, ensure_ascii=False)
        redis_client.setex(key, config.RETRIEVE_CACHE_TTL, json_str)
        log_info("检索缓存写入", f"成功写入检索缓存 key={key} TTL={config.RETRIEVE_CACHE_TTL}s")
    except redis.RedisError as e:
        log_error("检索缓存写入Redis失败", f"key={key} 写入异常", e)
    except Exception as e:
        log_error("检索缓存序列化失败", f"key={key} 文档转JSON出错", e)

def delete_retrieve_cache(query: str):
    """主动删除指定问句检索缓存（脏缓存清理专用）"""
    redis_client = _get_redis_client()
    if redis_client is None or not config.ENABLE_RAG_CACHE:
        return
    key = f"{config.RETRIEVE_CACHE_PREFIX}{get_query_hash(query)}"
    try:
        redis_client.delete(key)
        log_warn("检索缓存清理", f"清除脏检索缓存 key={key}")
    except redis.RedisError as e:
        log_error("检索缓存删除失败", f"key={key} 删除异常", e)

# ===================== 会话问答缓存读写 =====================
def get_chat_cache(session_id: str, query: str):
    """
    会话+问句双维度问答缓存读取
    隔离不同用户会话，细分Redis/序列化异常
    """
    redis_client = _get_redis_client()
    cache_switch = config.ENABLE_RAG_CACHE
    hash_q = get_query_hash(query)
    key = f"{config.CHAT_CACHE_PREFIX}{session_id}:{hash_q}"

    if not cache_switch:
        log_debug("问答缓存读取", f"缓存开关关闭，跳过读取 key={key}")
        return None
    if redis_client is None:
        log_warn("问答缓存读取", f"Redis未初始化，禁用缓存读取 key={key}")
        return None

    # Redis读取异常
    try:
        raw_data = redis_client.get(key)
    except redis.RedisError as e:
        err_msg = f"Redis读取问答缓存失败 key={key}"
        log_error("问答缓存Redis异常", err_msg, e)
        raise RedisStorageException(err_msg, e)

    if raw_data is None:
        log_debug("问答缓存读取", f"会话{session_id}未命中问答缓存 key={key}")
        return None

    # JSON脏数据捕获
    try:
        cache_obj = json.loads(raw_data)
        reply = cache_obj["reply"]
    except (json.JSONDecodeError, KeyError) as e:
        err_msg = f"问答缓存key={key}数据格式损坏"
        log_error("问答缓存脏数据", err_msg, e)
        redis_client.delete(key)
        raise CacheSerializeException(err_msg, e)

    log_info("问答缓存读取", f"会话{session_id}完全命中问答缓存 key={key}")
    return reply

def set_chat_cache(session_id: str, query: str, reply: str):
    """写入完整问答缓存，写入失败仅日志，不抛出异常影响对话主流程"""
    redis_client = _get_redis_client()
    cache_switch = config.ENABLE_RAG_CACHE
    hash_q = get_query_hash(query)
    key = f"{config.CHAT_CACHE_PREFIX}{session_id}:{hash_q}"
    store_data = json.dumps({"query": query, "reply": reply}, ensure_ascii=False)

    if not cache_switch:
        log_debug("问答缓存写入", f"缓存开关关闭，跳过写入 key={key}")
        return
    if redis_client is None:
        log_warn("问答缓存写入", f"Redis未初始化，无法写入 key={key}")
        return

    try:
        redis_client.setex(key, config.CHAT_CACHE_TTL, store_data)
        log_info("问答缓存写入", f"会话{session_id}写入问答缓存 key={key} TTL={config.CHAT_CACHE_TTL}s")
    except redis.RedisError as e:
        log_error("问答缓存Redis写入失败", f"key={key}", e)
    except Exception as e:
        log_error("问答缓存序列化失败", f"key={key}", e)

# ===================== 全局缓存清理 =====================
def clear_all_rag_cache():
    """清空全部RAG两类缓存，捕获Redis批量删除异常"""
    redis_client = _get_redis_client()
    if redis_client is None:
        log_warn("缓存全局清理", "Redis未连接，无法清空缓存")
        return 0

    try:
        retrieve_keys = redis_client.keys(f"{config.RETRIEVE_CACHE_PREFIX}*")
        chat_keys = redis_client.keys(f"{config.CHAT_CACHE_PREFIX}*")
        all_keys = retrieve_keys + chat_keys
        if all_keys:
            redis_client.delete(*all_keys)
            log_info("缓存全局清理", f"成功清空RAG缓存，共删除{len(all_keys)}条key")
        return len(all_keys)
    except redis.RedisError as e:
        log_error("缓存全局清理失败", "Redis批量删除缓存异常", e)
        return 0

# ===================== Document序列化/反序列化工具 =====================
def serialize_docs(docs):
    """Document对象转可序列化字典，容错metadata为None"""
    res = []
    for doc in docs:
        res.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata or {}
        })
    return res

def deserialize_docs(doc_dict_list):
    """
    序列化字典还原Document，精细化异常捕获
    单条文档解析失败自动跳过，不整体报错
    """
    docs = []
    if not isinstance(doc_dict_list, list):
        log_warn("文档反序列化", "缓存文档数据非数组格式，返回空列表")
        return docs

    for idx, d in enumerate(doc_dict_list):
        try:
            page_content = d.get("page_content") or ""
            metadata = d.get("metadata") or {}
            if not page_content.strip():
                log_warn("文档反序列化", f"缓存第{idx}条文档内容为空，跳过")
                continue
            docs.append(Document(page_content=page_content, metadata=metadata))
        except Exception as e:
            log_warn("文档反序列化单条失败", f"缓存第{idx}条文档解析异常，跳过本条", e)
    return docs