# ===================== 模型配置 =====================
LLM_MODEL = "qwen2:7b"
EMBED_MODEL = "nomic-embed-text"
LLM_TEMP = 0.1

# ===================== 文档分块配置 =====================
CHUNK_SIZE = 120
CHUNK_OVERLAP = 15
SEPARATORS = ["\n\n", "\n", "。", "，", " "]

# ===================== 向量库配置 =====================
CHROMA_PERSIST_PATH = "./chroma_db"
RETRIEVE_TOP_K = 3
SIMILARITY_THRESHOLD = 0.6  # 相似度过滤阈值，低于则丢弃文档

# ===================== Agent反思配置 =====================
MAX_REFLECT_TIMES = 2

# ===================== 会话配置 =====================
DEFAULT_SESSION_ID = "user_001"

# ===================== BM25关键词检索配置 =====================
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

# ===================== 缓存配置 =====================
# 总开关：是否启用缓存
ENABLE_RAG_CACHE = True
# 检索文档缓存过期时间 单位：秒 30分钟
RETRIEVE_CACHE_TTL = 10
# 问答完整结果缓存过期时间 单位：秒 2小时
CHAT_CACHE_TTL = 10
# 缓存key前缀区分两层缓存
RETRIEVE_CACHE_PREFIX = "rag:ret:"
CHAT_CACHE_PREFIX = "rag:chat:"
# 相同问题哈希时忽略空格/换行，统一预处理
CACHE_NORMALIZE_WHITESPACE = True

# ===================== 日志配置 =====================
# 日志级别：DEBUG/INFO/WARNING/ERROR
LOG_LEVEL = "INFO"
# 是否将日志写入本地文件
LOG_TO_FILE = True
# 日志文件存放目录
LOG_SAVE_PATH = "./logs"
# 单个日志最大MB，滚动分割
LOG_MAX_SIZE_MB = 10
# 保留日志文件数量
LOG_BACKUP_COUNT = 7

# ===================== 戏曲文献生成配置 =====================
# 文献输出独立根目录（根目录内按类型分子文件夹）
LITERATURE_ROOT_DIR = "./literature_output"
# 文献子目录：txt / pdf / md
LITERATURE_SUB_DIRS = {
    "txt": "txt",
    "pdf": "pdf",
    "md": "md"
}
# 文献生成默认流派/剧种
LITERATURE_DEFAULT_GENRE = "京剧"
# 文献生成默认篇幅（字数约）
LITERATURE_DEFAULT_LENGTH = 800

# ===================== 检索指标评测配置 =====================
# 评测测试问题集（真实业务问题 -> 期望命中关键词列表）
# 用于计算召回率、命中率、MRR、NDCG等指标
EVAL_QUESTION_SET = [
    {
        "query": "LangGraph的核心优势是什么",
        "expected_keywords": ["LangGraph", "图", "状态", "节点", "边"]
    },
    {
        "query": "Chroma向量库如何进行相似度检索",
        "expected_keywords": ["Chroma", "向量", "检索", "相似度", "嵌入"]
    },
    {
        "query": "RAG流程中文本分块的作用",
        "expected_keywords": ["分块", "chunk", "文本", "分割", "向量"]
    },
    {
        "query": "BM25关键词检索的原理",
        "expected_keywords": ["BM25", "关键词", "词频", "文档", "检索"]
    },
    {
        "query": "Agent反思机制如何工作",
        "expected_keywords": ["反思", "Agent", "重试", "规划", "工具"]
    }
]
# 评测指标：Recall@K / HitRate@K / MRR@K / NDCG@K
EVAL_TOP_K_LIST = [1, 3, 5]
# 评测测试轮次（多次执行取均值，结果更稳定）
EVAL_ROUNDS = 3
# 评测报告单独保存路径
EVAL_REPORT_PATH = "./evaluation_report/report.md"