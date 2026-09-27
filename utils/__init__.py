import traceback
from functools import wraps
import json

def global_exception_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            print(f"【文件异常】文件不存在：{str(e)}")
        except ConnectionError:
            print("【连接异常】Ollama服务未启动，请先打开ollama客户端")
        except json.JSONDecodeError:
            print("【解析异常】模型输出非标准JSON，已自动重试一次")
            return func(*args, **kwargs)
        except Exception as e:
            print(f"【未知异常】{str(e)}")
            traceback.print_exc()
        return None
    return wrapper