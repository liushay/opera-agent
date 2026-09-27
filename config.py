# ===================== 模型配置 =====================
# 已本地部署：qwen2.5:3b（长上下文 + 资源占用低）、bge-m3（中文检索更强）
LLM_MODEL = "qwen2.5:3b"
EMBED_MODEL = "bge-m3"
LLM_TEMP = 0.1

# ===================== 即梦AI图像生成（火山引擎版） =====================
# 基于火山引擎"视觉智能 CV"服务，使用 volcengine-python-sdk 中的 VisualService。
# 需要先安装依赖：pip install volcengine-python-sdk
# 接口地址：https://visual.volcengineapi.com  Region: cn-north-1  Service: cv
# 流程：CVSync2AsyncSubmitTask 异步提交 -> 轮询 CVAsyncQueryTask -> 下载图片
JIMENG_ACCESS_KEY_ID = ""              # 火山引擎 AccessKey ID
JIMENG_SECRET_ACCESS_KEY = ""          # 火山引擎 Secret Access Key
JIMENG_IMAGE_ENABLED = False       # 是否启用即梦图像生成（True 启用，False 跳过图像生成）
JIMENG_API_BASE = "https://visual.volcengineapi.com"  # 火山视觉智能 API 地址
JIMENG_API_ACTION = "CVSync2AsyncSubmitTask"          # 提交异步任务 Action
JIMENG_API_VERSION = "2022-08-31"      # API 版本
JIMENG_MODEL = "jimeng_t2i_v31"        # 文生图 req_key 固定值
JIMENG_IMAGE_OUTPUT_DIR = "./literature_output/face_images"  # 生成图片保存目录（不存在自动创建）
JIMENG_IMAGE_TIMEOUT = 120             # 异步任务最大等待秒数

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
BM25_TOP_K = 7
# 向量检索召回条数
VECTOR_TOP_K = 7
# 混合后最终返回文档数量
HYBRID_FINAL_K = 10
# 分数权重：向量分数权重 / BM25分数权重（可动态调参）
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4
# 是否开启混合检索，False则降级为纯向量检索（兼容旧逻辑）
ENABLE_HYBRID_SEARCH = True

# ===================== Reranker精排配置 =====================
# 是否启用LLM精排（BM25+向量召回→融合→精排）
ENABLE_RERANKER = True
# 精排模型（默认使用主LLM）
RERANKER_MODEL = ""
# 精排返回文档数（默认与混合检索最终K一致）
RERANKER_TOP_K = 3

# ===================== 缓存配置 =====================
# 总开关：是否启用缓存
ENABLE_RAG_CACHE = True
# 检索文档缓存过期时间 单位：秒 30分钟
RETRIEVE_CACHE_TTL = 100
# 问答完整结果缓存过期时间 单位：秒 2小时
CHAT_CACHE_TTL = 100
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
# 【已改造】不再硬编码固定测评问题集。
# 测评问题改为动态生成：根据传入主题 + 参考文档知识库内容，自动生成测评问题集合。
# EVAL_QUESTION_SET 保留为空列表作为默认值，兼容旧代码读取（getattr/config.EVAL_QUESTION_SET）。
EVAL_QUESTION_SET = []
# 动态评测生成问题数量（默认）
EVAL_DYNAMIC_QUESTION_COUNT = 6
# 动态评测参考文档检索条数（用于生成问题的知识库文档数）
EVAL_DYNAMIC_REF_DOCS_K = 5
# 评测指标：Recall@K / HitRate@K / MRR@K / NDCG@K
EVAL_TOP_K_LIST = [1, 3, 5]
# 评测测试轮次（多次执行取均值，结果更稳定）
EVAL_ROUNDS = 3
# 评测报告单独保存路径
EVAL_REPORT_PATH = "./evaluation_report/report.md"
# 动态测评问题生成的 LLM 调用超时（秒）：缩短超时避免评测接口整体超时，失败立即规则降级
EVAL_QUESTION_GEN_TIMEOUT = 45
# 评测接口整体最大超时（秒）：动态问题生成 + 多轮检索评测超时保护
EVALUATION_RUN_TIMEOUT = 300
# ===================== LLM 超时配置 =====================
# 统一 LLM 调用超时（秒），所有 invoke_with_retry 使用此值
LLM_TIMEOUT = 120

# ===================== 文献生成接口配置 =====================
# 文献生成接口最大超时时间（秒），超过则返回明确错误，避免无限挂起
LITERATURE_GENERATE_TIMEOUT = 180
