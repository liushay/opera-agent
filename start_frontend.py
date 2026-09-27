# start_frontend.py 前端启动脚本
# 以标准方式启动Streamlit应用
# 用法：python start_frontend.py
import subprocess
import sys
import os

if __name__ == "__main__":
    print("=" * 60)
    print("[Streamlit] 前端启动中...")
    print("  地址：http://localhost:8501")
    print("  请确保后端服务已启动（http://127.0.0.1:8000）")
    print("=" * 60)

    # 切换到frontend目录启动（保证相对导入正确）
    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"]
    subprocess.run(cmd, cwd=frontend_dir)