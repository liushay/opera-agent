from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_session_history(session_id:str) -> RedisChatMessageHistory:
    # 只填写redis连接地址，禁止传入client对象
    # protocol=2 使用 RESP2 协议，兼容旧版 Redis 服务器（不支持 RESP3 HELLO 命令）
    history = RedisChatMessageHistory(
        session_id=session_id,
        url="redis://127.0.0.1:6379/0?protocol=2"
    )
    return history
