from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_session_history(session_id:str) -> RedisChatMessageHistory:
    # 只填写redis连接地址，禁止传入client对象
    history = RedisChatMessageHistory(
        session_id=session_id,
        redis_url="redis://127.0.0.1:6379/0"
    )
    return history