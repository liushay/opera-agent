from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
from langchain_ollama import ChatOllama
import config
from utils.logger import log_info, log_error
from utils.rag_exceptions import LLMModelException
import httpx

async def stream_llm_response(prompt:str) -> AsyncGenerator[str, None]:
    log_info("流式LLM", "开始发起Ollama流式生成请求")
    try:
        llm = ChatOllama(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMP,
            streaming=True
        )
        async for chunk in llm.astream(prompt):
            yield chunk.content
        log_info("流式LLM", "Ollama流式输出完成")
    except httpx.ConnectError as e:
        err_msg = f"无法连接Ollama服务，请启动ollama，模型：{config.LLM_MODEL}"
        log_error("流式LLM连接失败", err_msg, e)
        raise LLMModelException(err_msg, e)
    except httpx.TimeoutException as e:
        err_msg = f"Ollama调用超时，模型：{config.LLM_MODEL}"
        log_error("流式LLM超时", err_msg, e)
        raise LLMModelException(err_msg, e)
    except Exception as e:
        # 模型不存在、内部报错统一归类LLM异常
        err_msg = f"Ollama模型调用异常，模型名{config.LLM_MODEL}"
        log_error("流式LLM未知错误", err_msg, e)
        raise LLMModelException(err_msg, e)