import redis

redis_url = "redis://127.0.0.1:6379/0"
redis_client: redis.Redis | None = None

def init_redis():
    global redis_client
    redis_client = redis.Redis(host="127.0.0.1",port=6379,db=0,decode_responses=True,protocol=2)

def close_redis():
    global redis_client
    if redis_client is not None:
        redis_client.close()
        redis_client = None