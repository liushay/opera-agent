# api/routes/system_routes.py 系统管理接口路由
# 包含：健康检查 / 缓存管理 / 日志级别调整
import logging
import config
from fastapi import APIRouter

from api.schema import CommonResponse
from utils.cache_utils import clear_all_rag_cache
from utils.logger import global_logger, log_info

# 路由前缀
router = APIRouter(prefix="/api", tags=["系统管理"])


@router.get("/health", response_model=CommonResponse)
def health_check():
    """服务健康检查"""
    return CommonResponse(code=200, msg="服务正常", data={})


@router.post("/cache/clear", response_model=CommonResponse)
def clear_cache():
    """清空全部RAG缓存"""
    count = clear_all_rag_cache()
    return CommonResponse(
        code=200, msg=f"成功清空RAG缓存，共删除{count}条", data={"clear_count": count}
    )


@router.get("/cache/status", response_model=CommonResponse)
def cache_status():
    """查看缓存配置状态"""
    return CommonResponse(
        code=200,
        msg="缓存配置状态",
        data={
            "enable_cache": config.ENABLE_RAG_CACHE,
            "retrieve_ttl": config.RETRIEVE_CACHE_TTL,
            "chat_ttl": config.CHAT_CACHE_TTL,
        },
    )


@router.post("/log/level", response_model=CommonResponse)
def set_log_level_api(level: str):
    """动态调整日志级别"""
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    if level not in level_map:
        return CommonResponse(code=400, msg="级别仅支持 DEBUG/INFO/WARNING/ERROR", data={})
    global_logger.setLevel(level_map[level])
    log_info("日志配置", f"动态修改日志级别为{level}")
    return CommonResponse(code=200, msg=f"日志级别已切换至{level}", data={"current_level": level})