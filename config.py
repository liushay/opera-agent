# 模型配置
LLM_MODEL = "qwen2:7b"
EMBED_MODEL = "nomic-embed-text"
LLM_TEMP = 0.1

# 文档分块配置
CHUNK_SIZE = 120
CHUNK_OVERLAP = 15
SEPARATORS = ["\n\n", "\n", "。", "，", " "]

# 向量库配置
CHROMA_PERSIST_PATH = "./chroma_db"
RETRIEVE_TOP_K = 3
SIMILARITY_THRESHOLD = 0.6  # 相似度过滤阈值，低于则丢弃文档

# Agent反思配置
MAX_REFLECT_TIMES = 2

# 会话配置
DEFAULT_SESSION_ID = "user_001"

# BM25关键词检索召回条数
BM25_TOP_K = 3
# 向量检索召回条数
VECTOR_TOP_K = 3
# 混合后最终返回文档数量
HYBRID_FINAL_K = 3
# 分数权重：向量分数权重 / BM25分数权重（可动态调参）
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4
# 是否开启混合检索，False则降级为纯向量检索（兼容旧逻辑）
ENABLE_HYBRID_SEARCH = True

# 总开关：是否启用缓存
ENABLE_RAG_CACHE = True
# 检索文档缓存过期时间 单位：秒 30分钟
RETRIEVE_CACHE_TTL = 1800
# 问答完整结果缓存过期时间 单位：秒 2小时
CHAT_CACHE_TTL = 7200
# 缓存key前缀区分两层缓存
RETRIEVE_CACHE_PREFIX = "rag:ret:"
CHAT_CACHE_PREFIX = "rag:chat:"
# 相同问题哈希时忽略空格/换行，统一预处理
CACHE_NORMALIZE_WHITESPACE = True