# start_backend.py 后端启动脚本
# 以标准方式启动FastAPI服务（避免reload导致的多进程缓存问题）
# 用法：python start_backend.py
import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("[RAG] 知识库后端服务启动中...")
    print("  地址：http://127.0.0.1:8000")
    print("  API文档：http://127.0.0.1:8000/docs")
    print("=" * 60)
    # 使用非reload模式，保证Agent实例在lifespan中只初始化一次
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )