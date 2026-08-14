from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import config
from api.schema import ChatRequest, MultiAgentRequest, CommonResponse
from agent.session_memory import get_session_history
from langchain_core.messages import AIMessage, HumanMessage
from agent.graph_base import build_agent_graph
from agent.multi_agent import build_multi_agent
from kb_manager.chroma_kb import kb
from utils.cache_utils import get_chat_cache, set_chat_cache, clear_all_rag_cache
from utils.logger import print_log, log_info
from utils.redis_client import init_redis, close_redis
from api.stream_response import stream_llm_response
from contextlib import asynccontextmanager
from utils.rag_exceptions import BaseRAGException
from utils.logger import log_error
from fastapi.responses import JSONResponse

# 1.限速器实例
limiter = Limiter(key_func=get_remote_address)

# 2.生命周期钩子
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_redis()
    print_log("系统启动", "Redis会话存储初始化成功")
    # Day13新增：启动时重建BM25索引
    kb.rebuild_full_bm25()
    print_log("系统启动", "混合检索BM25索引加载完成")
    # 初始化Agent实例（原startup事件逻辑移入lifespan，消除冲突）
    global single_agent, multi_agent
    single_agent = build_agent_graph()
    multi_agent = build_multi_agent()
    yield
    close_redis()
    print_log("系统关闭", "Redis连接已经关闭")

# 只新建一次FastAPI实例，挂载生命周期
app = FastAPI(title="本地Agent知识库后端‑完整版", lifespan=lifespan)

# 业务自定义RAG异常捕获
@app.exception_handler(BaseRAGException)
async def rag_exception_handler(request: Request, exc: BaseRAGException):
    log_error(f"业务异常{exc.code}", f"{exc.msg} 原始错误：{str(exc.origin_err)}", exc.origin_err)
    return JSONResponse(
        status_code=500,
        content={
            "code": exc.code,
            "msg": exc.msg,
            "detail": str(exc.origin_err) if exc.origin_err else ""
        }
    )

# 通用未知系统异常捕获
@app.exception_handler(Exception)
async def global_unknown_exception_handler(request: Request, exc: Exception):
    log_error("系统未知异常", f"接口全局捕获未处理错误：{str(exc)}", exc)
    return JSONResponse(
        status_code=500,
        content={
            "code": 9999,
            "msg": "服务内部未知错误，请查看日志排查",
            "detail": str(exc)
        }
    )

# 绑定限流组件至app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 延后初始化Agent（Agent实例已在lifespan启动阶段初始化）
single_agent = None
multi_agent = None


@app.post("/api/chat/normal", response_model=CommonResponse)
@limiter.limit("10/minute")
async def normal_chat(request: Request, chat_req: ChatRequest):
    try:
        cache_reply = get_chat_cache(chat_req.session_id, chat_req.query)
        if cache_reply is not None:
            print_log("问答缓存", f"会话{chat_req.session_id}完全命中问答缓存，直接返回")
            return CommonResponse(code=200, msg="请求成功(缓存命中)", data={"reply": cache_reply})

        history = get_session_history(chat_req.session_id)
        print_log("普通对话接口", f"会话:{chat_req.session_id} 用户提问:{chat_req.query}")
        msg_list = await history.aget_messages()
        agent_input = {
            "user_query": chat_req.query,
            "messages": msg_list,
            "tool_call": {},
            "tool_result": "",
            "reflect_times": 0
        }
        res = single_agent.invoke(agent_input)
        answer = res["messages"][-1].content
        # 保存用户本轮提问到会话记忆
        await history.aadd_messages([HumanMessage(content=chat_req.query)])
        # 保存AI回答到会话记忆
        await history.aadd_messages([AIMessage(content=answer)])
        # 写入问答缓存
        set_chat_cache(chat_req.session_id, chat_req.query, answer)

        return CommonResponse(code=200, msg="请求成功", data={"reply": answer})
    except RateLimitExceeded:
        raise HTTPException(status_code=429, detail="访问过于频繁，请稍后重试")
    except Exception as e:
        print_log("接口异常", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/multi_agent", response_model=CommonResponse)
@limiter.limit("8/minute")
async def run_multi_agent(request: Request, multi_req: MultiAgentRequest):
    try:
        # Day14新增：问答缓存优先判断（修正变量名）
        cache_reply = get_chat_cache(multi_req.session_id, multi_req.user_query)
        if cache_reply is not None:
            print_log(tag="问答缓存", content=f"会话{multi_req.session_id}完全命中问答缓存，直接返回")
            return CommonResponse(code=200, msg="请求成功(缓存命中)", data={"reply": cache_reply})

        print_log("多智能体接口", f"会话:{multi_req.session_id} 问题:{multi_req.user_query}")
        init_state = {
            "user_query": multi_req.user_query,
            "sub_task_list": [],
            "worker_result": [],
            "messages": [],
            "retry_times": 0,
            "need_retry": False
        }
        result = multi_agent.invoke(init_state)
        ans = result["messages"][-1].content
        set_chat_cache(multi_req.session_id, multi_req.user_query, ans)
        return CommonResponse(code=200, msg="多智能体执行完毕", data={"reply": ans})
    except RateLimitExceeded:
        raise HTTPException(status_code=429, detail="访问过于频繁，请稍后重试")
    except Exception as e:
        print_log("多Agent接口异常", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# 新增流式对话接口，SSE标准响应头 + 保存聊天历史
@app.post("/api/chat/stream")
@limiter.limit("15/minute")
async def stream_chat(request: Request, chat_req: ChatRequest):
    try:
        history = get_session_history(chat_req.session_id)
        msg_arr = await history.aget_messages()
        prompt_text = f"{msg_arr}\n用户新问题:{chat_req.query}"

        async def generator():
            # 保存用户消息到会话记忆
            await history.aadd_messages([HumanMessage(content=chat_req.query)])
            full_answer = ""
            async for chunk in stream_llm_response(prompt_text):
                full_answer += chunk
                yield f"data: {chunk}\n\n"
            # 流式结束之后存入Redis会话
            await history.aadd_messages([AIMessage(content=full_answer)])
            # 流式生成完成写入缓存
            set_chat_cache(chat_req.session_id, chat_req.query, full_answer)

        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
        return StreamingResponse(generator(), headers=headers)
    except RateLimitExceeded:
        raise HTTPException(status_code=429, detail="访问过于频繁，请稍后重试")
    except Exception as e:
        print_log("流式接口异常", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health_check():
    return CommonResponse(code=200, msg="服务正常", data={})


@app.post("/api/cache/clear")
def clear_cache():
    count = clear_all_rag_cache()
    return CommonResponse(code=200, msg=f"成功清空RAG缓存，共删除{count}条", data={"clear_count": count})

@app.get("/api/cache/status")
def cache_status():
    """查看缓存总开关状态"""
    return CommonResponse(
        code=200,
        msg="缓存配置状态",
        data={
            "enable_cache": config.ENABLE_RAG_CACHE,
            "retrieve_ttl": config.RETRIEVE_CACHE_TTL,
            "chat_ttl": config.CHAT_CACHE_TTL
        }
    )

@app.post("/api/log/level")
def set_log_level(level: str):
    import logging
    from utils.logger import global_logger
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    if level not in level_map:
        return CommonResponse(code=400, msg="级别仅支持 DEBUG/INFO/WARNING/ERROR", data={})
    global_logger.setLevel(level_map[level])
    log_info("日志配置", f"动态修改日志级别为{level}")
    return CommonResponse(code=200, msg=f"日志级别已切换至{level}", data={"current_level": level})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)