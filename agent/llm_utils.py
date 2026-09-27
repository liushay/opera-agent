# agent/llm_utils.py 统一LLM调用工具层
# 功能：
#   1. 所有LLM调用统一 timeout=120s，超时抛异常
#   2. 最大重试 2 次，禁止无限循环
#   3. 完整异常捕获，错误打印日志并向上抛出可读错误
#   4. 全链路日志埋点：开始 -> 调用LLM -> 完成
# 设计原则：
#   - 完全解耦，不依赖三层记忆结构，不改变任何业务模块对外接口
#   - 业务模块（literature/opera/agent）统一复用，根治"生成无限转圈/卡死/无报错阻塞"
import time
from typing import Any, List, Optional

import httpx
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

import config
from utils.logger import log_info, log_warn, log_error
from utils.rag_exceptions import LLMModelException

# LLM 调用超时（秒）—— 规范约定 120s
DEFAULT_LLM_TIMEOUT = getattr(config, "LLM_TIMEOUT", 120)
# 最大重试次数（规范约定 2 次）
MAX_LLM_RETRY = 2


def build_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    timeout: float = DEFAULT_LLM_TIMEOUT,
) -> ChatOllama:
    """
    统一构建 ChatOllama 实例（带超时配置）。
    - 默认使用 config.LLM_MODEL / config.LLM_TEMP
    - 超时通过 request_timeout 传入，避免长任务无响应卡死
    """
    return ChatOllama(
        model=model or getattr(config, "LLM_MODEL", "qwen2.5:3b"),
        temperature=temperature if temperature is not None else getattr(config, "LLM_TEMP", 0.1),
        request_timeout=timeout,
    )


def invoke_with_retry(
    prompts: List[str],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    timeout: float = DEFAULT_LLM_TIMEOUT,
    max_retry: int = MAX_LLM_RETRY,
    task_name: str = "LLM调用",
) -> str:
    """
    通用 LLM 调用封装：带超时 + 重试 + 全链路日志 + 异常捕获。
    Args:
        prompts: 提示词列表（多段时拼接为一条 HumanMessage）
        model: 模型名（默认 config.LLM_MODEL）
        temperature: 温度（默认 config.LLM_TEMP）
        timeout: 单次调用超时秒数（默认 120）
        max_retry: 最大重试次数（默认 2，共最多尝试 1+2=3 次）
        task_name: 日志埋点任务名（如"文献生成"、"戏词解剖"）
    Returns:
        模型返回的文本内容（去除首尾空白）
    Raises:
        LLMModelException: 超时或重试耗尽时抛出，带明确错误信息
    """
    prompt_text = "\n".join(prompts)
    log_info(task_name, f"开始调用LLM（重试上限{max_retry}次，超时{timeout}s）")

    llm = build_llm(model=model, temperature=temperature, timeout=timeout)

    attempt = 0
    while attempt <= max_retry:
        attempt += 1
        try:
            log_info(task_name, f"第{attempt}次调用LLM ...")
            start_ts = time.time()
            resp = llm.invoke([HumanMessage(content=prompt_text)])
            cost = round(time.time() - start_ts, 2)
            content = resp.content.strip()
            log_info(task_name, f"第{attempt}次调用成功，耗时{cost}s，内容长度{len(content)}字")
            return content
        except httpx.TimeoutException as e:
            log_warn(task_name, f"第{attempt}次调用超时（{timeout}s）：{e}")
        except httpx.ConnectError as e:
            log_warn(task_name, f"第{attempt}次调用Ollama连接失败：{e}")
        except Exception as e:
            log_warn(task_name, f"第{attempt}次调用异常：{e}")

        if attempt <= max_retry:
            log_info(task_name, f"准备重试（第{attempt + 1}次）...")

    err_msg = f"{task_name}失败：LLM调用在{max_retry + 1}次尝试后仍未成功（超时{timeout}s）"
    log_error(task_name, err_msg, None)
    raise LLMModelException(err_msg)


def safe_llm_call(
    prompts: List[str],
    task_name: str = "LLM调用",
    default_return: Any = "",
    **kwargs,
) -> Any:
    """
    安全的 LLM 调用：异常时返回默认值而非抛出（供非关键路径使用）。
    业务模块用此方法包裹可选增强逻辑，即使LLM失败也不阻断主流程。
    """
    try:
        return invoke_with_retry(prompts, task_name=task_name, **kwargs)
    except Exception as e:
        log_warn(task_name, f"LLM调用失败，返回默认值：{e}")
        return default_return