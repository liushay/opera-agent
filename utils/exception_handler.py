import functools
import asyncio
import httpx
import redis
from httpcore import TimeoutException

from utils.logger import log_error, log_warn
from utils.rag_exceptions import LLMModelException, RedisStorageException

def global_exception_handler(func):
    """
    统一全局异常捕获装饰器，同步/异步自动适配
    自动识别Ollama、Redis底层错误，封装对应业务异常
    """
    # 异步函数处理
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                # 120s超时控制
                return await asyncio.wait_for(func(*args, **kwargs), timeout=120)
            except asyncio.TimeoutError as e:
                err_msg = f"函数 {func.__name__} 执行超时(120s)"
                log_error("异步函数超时", err_msg, e)
                raise Exception("大模型执行超时，请简化你的问题") from e
            # Ollama 网络类错误
            except (httpx.ConnectError, TimeoutException, httpx.HTTPStatusError) as e:
                err_msg = f"Ollama服务调用异常，函数：{func.__name__}"
                log_error("Ollama连接异常", err_msg, e)
                raise LLMModelException(err_msg, e) from e
            # Redis 所有异常
            except redis.RedisError as e:
                err_msg = f"Redis读写异常，函数：{func.__name__}"
                log_error("Redis存储异常", err_msg, e)
                raise RedisStorageException(err_msg, e) from e
            # 其余所有未知异常
            except Exception as e:
                err_msg = f"异步函数 {func.__name__} 执行未知错误"
                log_error("异步函数通用异常", err_msg, e)
                raise  # 重新抛出，交给上层全局异常中间件

        return async_wrapper

    # 同步函数处理
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        # Ollama 网络类错误
        except (httpx.ConnectError, httpx.TimeoutError, httpx.HTTPStatusError) as e:
            err_msg = f"Ollama服务调用异常，函数：{func.__name__}"
            log_error("Ollama连接异常", err_msg, e)
            raise LLMModelException(err_msg, e) from e
        # Redis 所有异常
        except redis.RedisError as e:
            err_msg = f"Redis读写异常，函数：{func.__name__}"
            log_error("Redis存储异常", err_msg, e)
            raise RedisStorageException(err_msg, e) from e
        # 其余所有未知异常
        except Exception as e:
            err_msg = f"同步函数 {func.__name__} 执行未知错误"
            log_error("同步函数通用异常", err_msg, e)
            raise  # 重新抛出，交给上层处理

    return sync_wrapper