# api/schema.py API请求/响应数据模型定义
from pydantic import BaseModel, Field
from typing import Optional, List

import config


# ===================== 对话接口 =====================

# 普通对话请求
class ChatRequest(BaseModel):
    session_id: str = Field(..., description="会话ID")
    query: str = Field(..., description="用户问题")


# 多智能体专用请求体
class MultiAgentRequest(BaseModel):
    session_id: Optional[str] = "default_multi_session"
    user_query: str = Field(..., description="用户问题")


# ===================== 知识库管理接口 =====================

# 文档入库请求
class DocIngestRequest(BaseModel):
    file_path: str = Field(..., description="待入库文件的绝对/相对路径")


# 检索测试请求
class RetrieveRequest(BaseModel):
    query: str = Field(..., description="检索查询文本")
    top_k: Optional[int] = Field(config.RETRIEVE_TOP_K, description="返回文档数量")
    use_hybrid: Optional[bool] = Field(True, description="是否使用混合检索")


# ===================== 戏曲文献生成接口 =====================

class LiteratureGenerateRequest(BaseModel):
    genre: str = Field(config.LITERATURE_DEFAULT_GENRE, description="戏曲种类，如京剧、豫剧")
    theme: Optional[str] = Field("戏曲艺术特色与发展", description="文献主题")
    length: Optional[int] = Field(config.LITERATURE_DEFAULT_LENGTH, description="目标字数")
    formats: Optional[List[str]] = Field(
        ["txt", "pdf", "md"], description="输出格式列表，仅支持 txt/pdf/md"
    )
    title: Optional[str] = Field(None, description="自定义文件名（不含扩展名）")


# ===================== 检索指标评测接口 =====================

class EvaluationRequest(BaseModel):
    rounds: Optional[int] = Field(config.EVAL_ROUNDS, description="评测轮次，多次取均值")
    topic: Optional[str] = Field("", description="测评主题（动态生成问题集用，为空则尝试读取config）")
    reference_text: Optional[str] = Field("", description="参考文档内容（动态生成问题集用，可为空自动检索知识库）")
    question_count: Optional[int] = Field(None, description="动态生成问题数量（默认使用config.EVAL_DYNAMIC_QUESTION_COUNT）")


# ===================== 统一返回格式 =====================

class CommonResponse(BaseModel):
    code: int
    msg: str
    data: dict