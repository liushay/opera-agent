class BaseRAGException(Exception):
    """RAG基础业务异常，所有自定义异常父类"""
    code: int
    msg: str
    def __init__(self, msg: str, origin_err: Exception = None):
        self.msg = msg
        self.origin_err = origin_err
        super().__init__(self.msg)

# 1.向量库/检索异常
class VectorStoreException(BaseRAGException):
    code = 5001
    msg = "向量库操作失败"

# 2.BM25索引异常
class BM25IndexException(BaseRAGException):
    code = 5002
    msg = "BM25索引异常，索引未初始化或损坏"

# 3.Redis缓存/会话存储异常
class RedisStorageException(BaseRAGException):
    code = 5003
    msg = "Redis连接/读写缓存失败"

# 4.Ollama模型调用异常（嵌入/LLM对话）
class LLMModelException(BaseRAGException):
    code = 5004
    msg = "Ollama模型调用失败，服务未启动或模型不存在"

# 5.文档文件加载/分块异常
class DocProcessException(BaseRAGException):
    code = 5005
    msg = "文档加载、文本分块处理失败"

# 6.缓存脏数据/序列化异常
class CacheSerializeException(BaseRAGException):
    code = 5006
    msg = "缓存序列化/反序列化解析失败，脏缓存"

# 7.Agent流程执行异常
class AgentFlowException(BaseRAGException):
    code = 5007
    msg = "智能体执行流程异常，任务拆解/工具调用失败"