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

