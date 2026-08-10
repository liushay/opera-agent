import functools
import asyncio
from utils.logger import print_log

def global_exception_handler(func):
    @functools.wraps(func)
    async def async_wrapper(*args,**kwargs):
        try:
            return await asyncio.wait_for(func(*args,**kwargs),timeout=120)
        except asyncio.TimeoutError:
            print_log("函数捕获异常",f"{func.__name__} 请求执行超时")
            raise Exception("大模型执行超时，请简化你的问题")
        except Exception as e:
            print_log("函数捕获异常",f"{func.__name__} 报错：{str(e)}")
            raise e
    return async_wrapper