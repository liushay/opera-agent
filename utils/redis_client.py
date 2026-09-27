# redis_client.py
import redis
from utils.rag_exceptions import RedisStorageException
from utils.logger import log_error, log_info, log_warn

redis_url = "redis://127.0.0.1:6379/0"
redis_client: redis.Redis | None = None

def init_redis():
    global redis_client
    try:
        redis_client = redis.Redis(
            host="127.0.0.1",port=6379,db=0,
            decode_responses=True,protocol=2,
            socket_timeout=5
        )
        redis_client.ping() # 连通性检测
        log_info("Redis初始化", "Redis服务连接成功")
    except redis.ConnectionError as e:
        err = RedisStorageException("Redis服务连接失败，请检查Redis是否启动", e)
        log_error("Redis初始化失败", err.msg, e)
        raise err
    except Exception as e:
        err = RedisStorageException("Redis初始化未知错误", e)
        log_error("Redis初始化未知异常", err.msg, e)
        raise err

def close_redis():
    global redis_client
    if redis_client is not None:
        try:
            redis_client.close()
            log_info("Redis关闭", "Redis连接正常关闭")
        except Exception as e:
            log_warn("Redis关闭异常", "关闭连接时出现错误", e)
        redis_client = None