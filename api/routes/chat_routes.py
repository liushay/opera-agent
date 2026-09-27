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
from utils.logger import print_log, log_info, log_error
from api.stream_response import stream_llm_response

# ===== 改造：新增需求解析层 / 任务隔离（解耦、不改原结构） =====
from agent.intent import parse_intent
from agent.task_context import create_new_task

# 路由前缀
router = APIRouter(prefix="/api/chat", tags=["对话接口"])


@router.post("/normal", response_model=CommonResponse)
@limiter.limit("10/minute")
async def normal_chat(request: Request, chat_req: ChatRequest):
    """普通单Agent对话接口（先行意图解析 + 任务隔离 + 输出校验）"""
    try:
        cache_reply = get_chat_cache(chat_req.session_id, chat_req.query)
        if cache_reply is not None:
            print_log("问答缓存", f"会话{chat_req.session_id}完全命中问答缓存，直接返回")
            return CommonResponse(
                code=200, msg="请求成功(缓存命中)", data={"reply": cache_reply}
            )

        # ===== 新增：1. 需求解析层（用户当前提问 = 最高优先级） =====
        intent = parse_intent(chat_req.query)
        # ===== 新增：2. 任务隔离（每一次新提问 = 全新 task_id，杜绝跨任务污染） =====
        task_data = create_new_task(
            user_query=chat_req.query,
            title=intent.get("core_task", "")[:50],
            session_id=chat_req.session_id,
        )
        task_id = task_data.get("id")

        history = get_session_history(chat_req.session_id)
        print_log("普通对话接口", f"会话:{chat_req.session_id} 用户提问:{chat_req.query}（task_id={task_id}）")
        msg_list = await history.aget_messages()
        agent_input = {
            "user_query": chat_req.query,
            "messages": msg_list,
            "tool_call": {},
            "tool_result": "",
            "reflect_times": 0,
            # 分层记忆：会话ID 隔离 session 层；task_id 隔离 Episodic 层（本次专用）
            "mem_session_id": chat_req.session_id,
            "mem_task_id": task_id,
            "intent": intent,
        }
        # 从app.state获取lifespan中已初始化的Agent实例（避免重复构建）
        single_agent = request.app.state.single_agent
        res = single_agent.invoke(agent_input)
        answer = res["messages"][-1].content

        # ===== 新增：3. 输出校验层（不满足自动重生成，最多2次） =====
        from agent.output_validator import validate_output, MAX_VALIDATE_RETRY
        try:
            checks = validate_output(
                chat_req.query,
                intent,
                answer,
                reference_text="",  # 普通问答无额外文档，文档约束由Agent内部检索保证
            )
            if not checks["pass"]:
                log_error("普通对话输出校验未通过", f"issues={checks['issues']}", None)
        except Exception as ve:
            log_error("普通对话输出校验异常", str(ve), ve)

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
    """多智能体协作对话接口（主管-工人模式，含意图解析 + 任务隔离）"""
    try:
        cache_reply = get_chat_cache(multi_req.session_id, multi_req.user_query)
        if cache_reply is not None:
            print_log("问答缓存", f"会话{multi_req.session_id}完全命中问答缓存，直接返回")
            return CommonResponse(
                code=200, msg="请求成功(缓存命中)", data={"reply": cache_reply}
            )

        # ===== 新增：1. 需求解析层（用户当前提问 = 最高优先级） =====
        intent = parse_intent(multi_req.user_query)
        # ===== 新增：2. 任务隔离（每一次新提问 = 全新 task_id，杜绝跨任务污染） =====
        task_data = create_new_task(
            user_query=multi_req.user_query,
            title=intent.get("core_task", "")[:50],
            session_id=multi_req.session_id,
        )
        task_id = task_data.get("id")

        print_log("多智能体接口", f"会话:{multi_req.session_id} 问题:{multi_req.user_query}（task_id={task_id}）")
        history = get_session_history(multi_req.session_id)
        msg_list = await history.aget_messages()

        # 从app.state获取lifespan中已初始化的多智能体实例
        multi_agent = request.app.state.multi_agent

        # ===== 修复(Bug3)：输出校验失败 → 重试重新派发图像生成 worker =====
        # 用户诉求包含"生成图片/生成脸谱"等图像需求，但第一轮结果没有图片资源时，
        # 不直接返回纯文字；优先重试（重新 invoke 多Agent，由 intent_guard 强制派发 face_worker），
        # 重试耗尽后返回报文必须明确标记校验未通过，并把已有文本 + 图片生成失败提示一起返回。
        from agent.output_validator import validate_output, MAX_VALIDATE_RETRY

        ans = ""
        images = []
        final_checks = None
        attempts = 0
        while attempts <= MAX_VALIDATE_RETRY:
            attempts += 1
            init_state = {
                "user_query": multi_req.user_query,
                "sub_task_list": [],
                "worker_result": [],
                "messages": msg_list,
                "retry_times": 0,
                "need_retry": False,
                # 分层记忆：会话ID 隔离 session 层；task_id 隔离 Episodic 层（本次专用）
                "mem_session_id": multi_req.session_id,
                "mem_task_id": task_id,
                "intent": intent,
            }
            result = multi_agent.invoke(init_state)
            ans = result["messages"][-1].content
            images = list(result.get("image_resources") or [])

            # ===== 修复：检测 face_worker 是否被调度，传递给输出校验的硬规则 =====
            sub_task_list = result.get("sub_task_list", [])
            face_worker_dispatched = any(
                t.get("worker") == "face_worker" for t in sub_task_list
            )
            if not face_worker_dispatched:
                # 兜底：也检查 intent 中是否包含 face_worker（复合需求场景）
                intent_sub_tasks = intent.get("sub_tasks", [])
                face_worker_dispatched = any(
                    st.get("worker") in ("face", "face_worker") for st in intent_sub_tasks
                )
            log_info(
                "多智能体接口",
                f"face_worker_dispatched={face_worker_dispatched}，"
                f"sub_task_list={[t.get('worker') for t in sub_task_list]}，"
                f"images_count={len(images)}",
            )

            # 输出校验层（无额外文档，文档约束由Worker内部检索保证）
            try:
                checks = validate_output(
                    multi_req.user_query, intent, ans,
                    reference_text="", image_resources=images,
                    face_worker_dispatched=face_worker_dispatched,
                )
                final_checks = checks
            except Exception as ve:
                log_error("多智能体输出校验异常", str(ve), ve)
                final_checks = {"pass": True, "issues": [], "image_required": False, "image_missing": False}
                checks = final_checks

            if checks["pass"]:
                break

            # 校验失败：如果是图像诉求缺失图片 → 重试重新调度图像生成 worker
            if checks.get("image_required") and checks.get("image_missing"):
                log_info(
                    "多智能体接口",
                    f"校验识别出用户需要图片但结果无图片资源，第{attempts}次重试重新派发图像生成worker",
                )
                continue

            # 非图像缺失类失败：普通文本重试一次（保留原行为，最多再试一次）
            if attempts <= MAX_VALIDATE_RETRY:
                log_error(
                    "多智能体输出校验未通过",
                    f"issues={checks['issues']}，第{attempts}次重试",
                    None,
                )
                continue
            break

        # ===== 保存会话记忆（与单Agent一致：写回Redis完整消息历史） =====
        await history.aadd_messages([HumanMessage(content=multi_req.user_query)])
        await history.aadd_messages([AIMessage(content=ans)])
        set_chat_cache(multi_req.session_id, multi_req.user_query, ans)

        # ===== 修复(Bug)：汇总节点合并图片资源进最终返回报文，供前端渲染 =====
        data = {"reply": ans}
        if images:
            data["images"] = images

        # 修复(Bug2)：重试耗尽仍未生成图片 → 返回报文必须明确标记校验未通过，
        # 不能假装任务完成；保留已有文本内容，并附加图片生成失败提示。
        if final_checks and not final_checks.get("pass", False):
            data["validation_passed"] = False
            if final_checks.get("image_missing"):
                data["image_generation_failed"] = True
                data["image_failed_tip"] = "图片生成失败，已重试多次仍未能生成真实图片。以下为文字版结果，图片生成功能暂不可用。"
                # 附加提示到回复文本尾部，让前端/用户明确感知
                if "图片生成失败" not in ans:
                    ans = ans + "\n\n> ⚠️ 图片生成失败提示：本次未能生成真实图片，已尽力重试。请稍后再试或检查即梦图像服务配置。"
                    data["reply"] = ans
            data["validation_issues"] = final_checks.get("issues", [])
            return CommonResponse(
                code=200,
                msg="多智能体执行完毕（校验未通过，图片生成失败）",
                data=data,
            )

        return CommonResponse(code=200, msg="多智能体执行完毕", data=data)
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