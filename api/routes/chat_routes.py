# api/routes/chat_routes.py 对话接口路由
# 包含：普通对话 / 多智能体 / 流式对话
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from slowapi.errors import RateLimitExceeded

from api.schema import ChatRequest, MultiAgentRequest, CommonResponse
from api.limiter import limiter
from agent.session_memory import get_session_history
from langchain_core.messages import AIMessage, HumanMessage
from utils.cache_utils import get_chat_cache, set_chat_cache
from utils.logger import print_log, log_error
from api.stream_response import stream_llm_response

# 路由前缀
router = APIRouter(prefix="/api/chat", tags=["对话接口"])


@router.post("/normal", response_model=CommonResponse)
@limiter.limit("10/minute")
async def normal_chat(request: Request, chat_req: ChatRequest):
    """普通单Agent对话接口"""
    try:
        cache_reply = get_chat_cache(chat_req.session_id, chat_req.query)
        if cache_reply is not None:
            print_log("问答缓存", f"会话{chat_req.session_id}完全命中问答缓存，直接返回")
            return CommonResponse(
                code=200, msg="请求成功(缓存命中)", data={"reply": cache_reply}
            )

        history = get_session_history(chat_req.session_id)
        print_log("普通对话接口", f"会话:{chat_req.session_id} 用户提问:{chat_req.query}")
        msg_list = await history.aget_messages()
        agent_input = {
            "user_query": chat_req.query,
            "messages": msg_list,
            "tool_call": {},
            "tool_result": "",
            "reflect_times": 0,
            # 分层记忆：将当前会话ID传入时序记忆（API参数保持不变）
            "mem_session_id": chat_req.session_id,
        }
        # 从app.state获取lifespan中已初始化的Agent实例（避免重复构建）
        single_agent = request.app.state.single_agent
        res = single_agent.invoke(agent_input)
        answer = res["messages"][-1].content
        # 保存会话记忆
        await history.aadd_messages([HumanMessage(content=chat_req.query)])
        await history.aadd_messages([AIMessage(content=answer)])
        # 写入问答缓存
        set_chat_cache(chat_req.session_id, chat_req.query, answer)

        return CommonResponse(code=200, msg="请求成功", data={"reply": answer})
    except RateLimitExceeded:
        raise HTTPException(status_code=429, detail="访问过于频繁，请稍后重试")
    except Exception as e:
        log_error("普通对话接口异常", str(e), e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multi_agent", response_model=CommonResponse)
@limiter.limit("8/minute")
async def run_multi_agent(request: Request, multi_req: MultiAgentRequest):
    """多智能体协作对话接口（主管-工人模式）"""
    try:
        cache_reply = get_chat_cache(multi_req.session_id, multi_req.user_query)
        if cache_reply is not None:
            print_log("问答缓存", f"会话{multi_req.session_id}完全命中问答缓存，直接返回")
            return CommonResponse(
                code=200, msg="请求成功(缓存命中)", data={"reply": cache_reply}
            )

        print_log("多智能体接口", f"会话:{multi_req.session_id} 问题:{multi_req.user_query}")
        init_state = {
            "user_query": multi_req.user_query,
            "sub_task_list": [],
            "worker_result": [],
            "messages": [],
            "retry_times": 0,
            "need_retry": False,
        }
        # 从app.state获取lifespan中已初始化的多智能体实例
        multi_agent = request.app.state.multi_agent
        result = multi_agent.invoke(init_state)
        ans = result["messages"][-1].content
        set_chat_cache(multi_req.session_id, multi_req.user_query, ans)
        return CommonResponse(code=200, msg="多智能体执行完毕", data={"reply": ans})
    except RateLimitExceeded:
        raise HTTPException(status_code=429, detail="访问过于频繁，请稍后重试")
    except Exception as e:
        log_error("多Agent接口异常", str(e), e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
@limiter.limit("15/minute")
async def stream_chat(request: Request, chat_req: ChatRequest):
    """流式对话接口（SSE协议），实时推送token"""
    try:
        history = get_session_history(chat_req.session_id)
        msg_arr = await history.aget_messages()
        prompt_text = f"{msg_arr}\n用户新问题:{chat_req.query}"

        async def generator():
            # 保存用户消息
            await history.aadd_messages([HumanMessage(content=chat_req.query)])
            full_answer = ""
            async for chunk in stream_llm_response(prompt_text):
                full_answer += chunk
                yield f"data: {chunk}\n\n"
            # 保存AI回答 + 写缓存
            await history.aadd_messages([AIMessage(content=full_answer)])
            set_chat_cache(chat_req.session_id, chat_req.query, full_answer)

        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        return StreamingResponse(generator(), headers=headers)
    except RateLimitExceeded:
        raise HTTPException(status_code=429, detail="访问过于频繁，请稍后重试")
    except Exception as e:
        log_error("流式接口异常", str(e), e)
        raise HTTPException(status_code=500, detail=str(e))