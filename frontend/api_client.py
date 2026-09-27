# frontend/api_client.py 后端API客户端
# 封装所有FastAPI接口的HTTP调用，供Streamlit前端使用
import requests
from typing import Dict, Any, List

# 后端服务地址（可通过环境变量覆盖）
BASE_URL = "http://127.0.0.1:8000"


class ApiClient:
    """统一后端API客户端，封装所有接口调用"""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip("/")
        # 默认会话（前端用户ID）
        self.session_id = "streamlit_user"

    # ===================== HTTP基础封装 =====================

    def _post(self, path: str, data: dict = None) -> Dict[str, Any]:
        """POST请求封装"""
        try:
            resp = requests.post(
                f"{self.base_url}{path}",
                json=data or {},
                timeout=600,  # 文献生成/评测可能耗时较长
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as e:
            return {"code": -1, "msg": f"无法连接后端服务：{self.base_url}", "data": {}}
        except requests.exceptions.Timeout as e:
            return {"code": -1, "msg": f"请求超时：{path}", "data": {}}
        except requests.exceptions.HTTPError as e:
            return {"code": -1, "msg": f"HTTP错误：{e}", "data": {}}
        except Exception as e:
            return {"code": -1, "msg": f"请求异常：{str(e)}", "data": {}}

    def _get(self, path: str, params: dict = None) -> Dict[str, Any]:
        """GET请求封装"""
        try:
            resp = requests.get(
                f"{self.base_url}{path}",
                params=params or {},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError as e:
            return {"code": -1, "msg": f"无法连接后端服务：{self.base_url}", "data": {}}
        except requests.exceptions.Timeout as e:
            return {"code": -1, "msg": f"请求超时：{path}", "data": {}}
        except requests.exceptions.HTTPError as e:
            return {"code": -1, "msg": f"HTTP错误：{e}", "data": {}}
        except Exception as e:
            return {"code": -1, "msg": f"请求异常：{str(e)}", "data": {}}

    # ===================== 对话接口 =====================

    def chat_normal(self, query: str) -> Dict[str, Any]:
        """普通Agent对话"""
        return self._post("/api/chat/normal", {
            "session_id": self.session_id,
            "query": query,
        })

    def chat_multi_agent(self, query: str) -> Dict[str, Any]:
        """多智能体对话"""
        return self._post("/api/chat/multi_agent", {
            "session_id": self.session_id,
            "user_query": query,
        })

    # ===================== 知识库接口 =====================

    def kb_ingest(self, file_path: str) -> Dict[str, Any]:
        """文档入库（按路径）"""
        return self._post("/api/kb/ingest", {"file_path": file_path})

    def kb_retrieve(self, query: str) -> Dict[str, Any]:
        """知识库检索测试"""
        return self._post("/api/kb/retrieve", {"query": query})

    def kb_stats(self) -> Dict[str, Any]:
        """知识库统计"""
        return self._get("/api/kb/stats")

    def kb_clear(self) -> Dict[str, Any]:
        """清空知识库"""
        return self._post("/api/kb/clear", {})

    def kb_upload(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """上传文件入库"""
        try:
            resp = requests.post(
                f"{self.base_url}/api/kb/upload",
                files={"file": (filename, file_bytes)},
                timeout=600,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"code": -1, "msg": f"上传失败：{str(e)}", "data": {}}

    # ===================== 文献生成接口 =====================

    def literature_generate(
        self,
        genre: str,
        theme: str,
        length: int,
        formats: List[str],
        title: str = None,
    ) -> Dict[str, Any]:
        """生成戏曲文献"""
        return self._post("/api/literature/generate", {
            "genre": genre,
            "theme": theme,
            "length": length,
            "formats": formats,
            "title": title,
        })

    def literature_list(self) -> Dict[str, Any]:
        """列出文献文件"""
        return self._get("/api/literature/list")

    def literature_read(self, path: str) -> Dict[str, Any]:
        """读取文献内容"""
        return self._get("/api/literature/read", {"path": path})

    def literature_download_url(self, path: str) -> str:
        """获取文献下载地址"""
        return f"{self.base_url}/api/literature/download?path={path}"

    def literature_status(self, task_id: str = None) -> Dict[str, Any]:
        """查询文献生成任务状态（前端可轮询感知是否正在生成）"""
        if task_id:
            return self._get(f"/api/literature/status/{task_id}")
        return self._get("/api/literature/status")

    # ===================== 评测接口 =====================

    def evaluation_run(self, rounds: int = 1) -> Dict[str, Any]:
        """执行检索指标评测"""
        return self._post("/api/evaluation/run", {"rounds": rounds})

    def evaluation_report(self) -> Dict[str, Any]:
        """读取评测报告"""
        return self._get("/api/evaluation/report")

    # ===================== 戏曲科普接口 =====================

    def lyrics_annotate(self, lyrics: str) -> Dict[str, Any]:
        """戏词解剖"""
        return self._post("/api/opera/lyrics/annotate", {"lyrics": lyrics})

    def character_list(self) -> Dict[str, Any]:
        """人物列表"""
        return self._get("/api/opera/character/list")

    def character_chat(self, character_id: str, message: str) -> Dict[str, Any]:
        """戏中人对谈"""
        return self._post("/api/opera/character/chat", {
            "character_id": character_id,
            "message": message,
            "session_id": self.session_id,
        })

    def quiz_generate(self, topic: str = "混合", difficulty: str = "入门") -> Dict[str, Any]:
        """知识闯关出题"""
        return self._post("/api/opera/quiz/generate", {
            "topic": topic,
            "difficulty": difficulty,
            "session_id": self.session_id,
        })

    def quiz_topics(self) -> Dict[str, Any]:
        """知识闯关主题"""
        return self._post("/api/opera/quiz/topics")

    def quiz_check(self, question_id: str, user_answer: str,
                   correct_answer: str, explanation: str) -> Dict[str, Any]:
        """知识闯关判题"""
        return self._post("/api/opera/quiz/check", {
            "question_id": question_id,
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "explanation": explanation,
            "session_id": self.session_id,
        })

    def course_generate(self, topic: str, days: int = 7,
                        level: str = "入门") -> Dict[str, Any]:
        """学戏路线生成"""
        return self._post("/api/opera/course/generate", {
            "topic": topic,
            "days": days,
            "level": level,
            "session_id": self.session_id,
        })

    def course_progress(self) -> Dict[str, Any]:
        """学戏路线进度"""
        return self._post("/api/opera/course/progress", {
            "session_id": self.session_id,
        })

    def face_generate(self, preferences: str) -> Dict[str, Any]:
        """脸谱画像生成"""
        return self._post("/api/opera/face/generate", {
            "preferences": preferences,
            "session_id": self.session_id,
        })

    # ===================== 系统管理接口 =====================

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return self._get("/api/health")

    def cache_status(self) -> Dict[str, Any]:
        """缓存状态"""
        return self._get("/api/cache/status")

    def cache_clear(self) -> Dict[str, Any]:
        """清空缓存"""
        return self._post("/api/cache/clear", {})


# 全局单例客户端
api_client = ApiClient()