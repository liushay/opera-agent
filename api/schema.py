from pydantic import BaseModel
from typing import Optional

# 普通对话请求
class ChatRequest(BaseModel):
    session_id: str
    query: str

# 多智能体专用请求体
class MultiAgentRequest(BaseModel):
    session_id: Optional[str] = "default_multi_session"
    user_query: str

# 统一返回格式
class CommonResponse(BaseModel):
    code: int
    msg: str
    data: dict