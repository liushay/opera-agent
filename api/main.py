# api/main.py FastAPI 后端主入口
# 负责：应用初始化 / 全局异常处理 / 路由注册
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager
import os
from pathlib import Path

from api.limiter import limiter
from api.routes.chat_routes import router as chat_router
from api.routes.kb_routes import router as kb_router
from api.routes.literature_routes import router as literature_router
from api.routes.evaluation_routes import router as evaluation_router
from api.routes.system_routes import router as system_router
from api.routes.opera_routes import router as opera_router
from agent.graph_base import build_agent_graph
from agent.multi_agent import build_multi_agent
from rag.vectorstore import kb
from literature.generator import literature_generator
from utils.logger import print_log, log_error
from utils.redis_client import init_redis, close_redis
from utils.rag_exceptions import BaseRAGException
from agent.memory import memory_manager
from agent.memory.session_memory import session_temporal_memory


# ===================== 生命周期管理 =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动/关闭生命周期钩子"""
    # ---------- 启动阶段 ----------
    # 1. 初始化Redis（缓存与会话存储）
    init_redis()
    print_log("系统启动", "Redis会话存储初始化成功")

    # 2. 重建BM25关键词索引
    kb.rebuild_full_bm25()
    print_log("系统启动", "混合检索BM25索引加载完成")

    # 3. 构建Agent实例并存入app.state（路由层从中获取，避免重复构建）
    app.state.single_agent = build_agent_graph()
    app.state.multi_agent = build_multi_agent()
    print_log("系统启动", "单Agent和多Agent实例初始化完成")

    # 4. 确认文献输出目录
    literature_generator._ensure_root_dirs()
    print_log("系统启动", "文献输出目录就绪")

    # 5. 初始化分层记忆模块（清理过期会话日志）
    try:
        cleaned = session_temporal_memory.cleanup_expired()
        print_log("系统启动", f"分层记忆模块就绪，三层记忆统计：{memory_manager.get_stats()}")
        if cleaned > 0:
            print_log("系统启动", f"已自动清理{cleaned}条过期会话时序记忆")
    except Exception as e:
        log_error("系统启动", f"分层记忆模块初始化异常：{str(e)}", e)

    yield

    # ---------- 关闭阶段 ----------
    close_redis()
    print_log("系统关闭", "Redis连接已经关闭")


# ===================== FastAPI 应用实例 =====================
app = FastAPI(
    title="本地Agent知识库后端-秋招工程版",
    description=(
        "RAG知识库问答 + 戏曲文献生成 + 检索指标评测一体化服务\n\n"
        "## 核心能力\n"
        "- **对话**：普通Agent / 多智能体协作 / 流式SSE对话\n"
        "- **知识库**：文档入库 / 混合检索 / 文件上传\n"
        "- **文献生成**：戏曲文献一键生成，支持txt/pdf/md三种格式\n"
        "- **指标评测**：召回率/命中率/MRR/NDCG量化测试\n"
        "- **戏曲科普**：戏词解剖室 / 戏中人对谈 / 知识闯关 / 学戏路线 / 脸谱画像"
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ===================== 全局异常处理 =====================


@app.exception_handler(BaseRAGException)
async def rag_exception_handler(request: Request, exc: BaseRAGException):
    """业务自定义异常捕获"""
    log_error(f"业务异常{exc.code}", f"{exc.msg} 原始错误：{str(exc.origin_err)}", exc.origin_err)
    return JSONResponse(
        status_code=500,
        content={
            "code": exc.code,
            "msg": exc.msg,
            "detail": str(exc.origin_err) if exc.origin_err else "",
        },
    )


@app.exception_handler(Exception)
async def global_unknown_exception_handler(request: Request, exc: Exception):
    """兜底全局未知异常捕获"""
    log_error("系统未知异常", f"接口全局捕获未处理错误：{str(exc)}", exc)
    return JSONResponse(
        status_code=500,
        content={
            "code": 9999,
            "msg": "服务内部未知错误，请查看日志排查",
            "detail": str(exc),
        },
    )


# ===================== 限流组件绑定 =====================
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ===================== 注册所有路由 =====================
app.include_router(chat_router)        # /api/chat/*
app.include_router(kb_router)          # /api/kb/*
app.include_router(literature_router)  # /api/literature/*
app.include_router(evaluation_router)  # /api/evaluation/*
app.include_router(system_router)      # /api/health /api/cache/* /api/log/*
app.include_router(opera_router)       # /api/opera/*


# ===================== 静态文件服务（脸谱图片等） =====================
# 将 literature_output 目录挂载为 /static，前端可访问生成的图片：
#   http://127.0.0.1:8000/static/face_images/xxx.png
_static_dir = Path(__file__).resolve().parent.parent / "literature_output"
_static_dir.mkdir(parents=True, exist_ok=True)
if not any(r.path == "/static" for r in app.routes):
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ===================== 启动入口 =====================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)