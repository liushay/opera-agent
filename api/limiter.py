# api/limiter.py 全局唯一限速器实例
# 所有路由模块统一从此处导入，确保与app.state.limiter为同一实例
from slowapi import Limiter
from slowapi.util import get_remote_address

# 全局唯一限速器：基于客户端IP限流
limiter = Limiter(key_func=get_remote_address)