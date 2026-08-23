# agent/memory/session_memory.py 会话时序记忆模块
# 功能：
#   1. 记录对话轨迹、思考、工具调用、结果、报错、修改记录
#   2. 附带时间戳
#   3. 自动归档清理过期日志
#   4. 与永久静态记忆、任务级记忆完全隔离
import json
import os
import uuid
import shutil
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from utils.logger import log_info, log_warn, log_error
from utils.rag_exceptions import BaseRAGException


class SessionTemporalMemoryException(BaseRAGException):
    """会话时序记忆异常"""
    code = 5103
    msg = "会话时序记忆操作失败"


class SessionTemporalMemory:
    """
    会话时序记忆管理器
    存储媒介：本地JSON目录（sessions/xxx.json），独立于向量库与其他记忆层
    记录类型（event_type）：
      - dialogue:   对话轨迹（用户/AI消息）
      - thinking:   思考过程
      - tool_call:  工具调用
      - tool_result: 工具结果
      - error:      报错信息
      - modify:     修改记录
    自动归档：当会话记录超过 MAX_EVENTS 或超过最大保存天数时自动归档
    """

    # 记录事件类型
    EVENT_TYPES = ("dialogue", "thinking", "tool_call", "tool_result", "error", "modify")

    def __init__(
        self,
        storage_dir: str = "./memory_store/sessions",
        archive_dir: str = "./memory_store/archives",
        max_events_per_session: int = 200,
        max_retention_days: int = 30,
    ):
        self.storage_dir = storage_dir
        self.archive_dir = archive_dir
        self.max_events_per_session = max_events_per_session
        self.max_retention_days = max_retention_days
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)

    # ===================== 路径工具 =====================

    def _session_path(self, session_id: str) -> str:
        """获取会话文件路径，校验session_id合法性"""
        session_id = str(session_id)
        if session_id in ("", ".", "..") or "/" in session_id or "\\" in session_id:
            raise SessionTemporalMemoryException(f"非法会话ID：{session_id}")
        return os.path.join(self.storage_dir, f"{session_id}.json")

    def _load_session(self, session_id: str) -> Dict[str, Any]:
        """加载会话数据"""
        path = self._session_path(session_id)
        if not os.path.exists(path):
            return {
                "id": session_id,
                "events": [],
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            err = SessionTemporalMemoryException(f"会话数据加载失败：{session_id}", e)
            log_error("会话时序记忆", err.msg, e)
            raise err

    def _save_session(self, session_id: str, data: Dict[str, Any]):
        """保存会话数据"""
        path = self._session_path(session_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            log_info("会话时序记忆", f"会话[{session_id}]保存成功")
        except Exception as e:
            err = SessionTemporalMemoryException(f"会话数据保存失败：{session_id}", e)
            log_error("会话时序记忆", err.msg, e)
            raise err

    # ===================== 事件记录 =====================

    def add_event(
        self,
        session_id: str,
        event_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        新增一条会话时序记忆事件
        session_id: 会话ID
        event_type: dialogue/thinking/tool_call/tool_result/error/modify
        content: 事件内容
        metadata: 附加元数据（如工具名、耗时等）
        """
        if event_type not in self.EVENT_TYPES:
            raise SessionTemporalMemoryException(
                f"非法事件类型：{event_type}，可选：{self.EVENT_TYPES}"
            )

        session = self._load_session(session_id)
        event_id = str(uuid.uuid4())
        now = datetime.now()
        event = {
            "id": event_id,
            "event_type": event_type,
            "content": content,
            "metadata": metadata or {},
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        }
        session["events"].append(event)
        session["updated_at"] = now.strftime("%Y-%m-%d %H:%M:%S")

        # 检查是否需要自动归档
        if len(session["events"]) > self.max_events_per_session:
            self._archive_and_trim(session_id, session)

        self._save_session(session_id, session)
        log_info("会话时序记忆", f"会话[{session_id}]新增[{event_type}]事件，id={event_id}")
        return event_id

    # ===================== 便捷方法 =====================

    def add_dialogue(self, session_id: str, role: str, content: str) -> str:
        """记录对话轨迹（role: user/ai）"""
        return self.add_event(
            session_id, "dialogue", content,
            metadata={"role": role},
        )

    def add_thinking(self, session_id: str, content: str) -> str:
        """记录思考过程"""
        return self.add_event(session_id, "thinking", content)

    def add_tool_call(self, session_id: str, tool_name: str, params: str) -> str:
        """记录工具调用"""
        return self.add_event(
            session_id, "tool_call", f"调用工具：{tool_name}",
            metadata={"tool_name": tool_name, "params": params},
        )

    def add_tool_result(self, session_id: str, tool_name: str, result: str) -> str:
        """记录工具结果"""
        return self.add_event(
            session_id, "tool_result", result,
            metadata={"tool_name": tool_name},
        )

    def add_error(self, session_id: str, error_msg: str, stack: str = "") -> str:
        """记录报错信息"""
        return self.add_event(
            session_id, "error", error_msg,
            metadata={"stack": stack},
        )

    def add_modify(self, session_id: str, description: str, target: str = "") -> str:
        """记录修改记录"""
        return self.add_event(
            session_id, "modify", description,
            metadata={"target": target},
        )

    # ===================== 查询 =====================

    def get_events(
        self,
        session_id: str,
        event_type: Optional[str] = None,
        limit: int = 50,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        查询会话时序记忆
        event_type: 按事件类型过滤
        limit: 返回最近N条
        start_time/end_time: 时间范围过滤（格式：YYYY-MM-DD HH:MM:SS）
        """
        session = self._load_session(session_id)
        events = session["events"]

        if event_type:
            events = [e for e in events if e["event_type"] == event_type]
        if start_time:
            events = [e for e in events if e["timestamp"] >= start_time]
        if end_time:
            events = [e for e in events if e["timestamp"] <= end_time]

        # 返回最近limit条
        return events[-limit:]

    def get_recent_dialogue(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的对话记录"""
        return self.get_events(session_id, event_type="dialogue", limit=limit)

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """获取会话概要信息"""
        session = self._load_session(session_id)
        events = session["events"]
        info = {
            "id": session["id"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "event_count": len(events),
            "dialogue_count": sum(1 for e in events if e["event_type"] == "dialogue"),
            "tool_call_count": sum(1 for e in events if e["event_type"] == "tool_call"),
            "error_count": sum(1 for e in events if e["event_type"] == "error"),
            "modify_count": sum(1 for e in events if e["event_type"] == "modify"),
        }
        return info

    def _tokenize(self, text: str) -> List[str]:
        """
        对查询/文本进行多粒度分词：
        1. 连续英文/数字为一个token
        2. 中文按单字和相邻二元组拆分（提升短查询命中率）
        """
        import re
        tokens = set()
        # 英文/数字整体
        for m in re.findall(r"[a-zA-Z0-9]+", text.lower()):
            tokens.add(m.lower())
        # 中文部分：单字 + 相邻二元组
        cn_chars = re.findall(r"[\u4e00-\u9fff]", text)
        if cn_chars:
            tokens.update(cn_chars)
            for i in range(len(cn_chars) - 1):
                tokens.add(cn_chars[i] + cn_chars[i + 1])
        return list(tokens)

    def search_for_agent(self, query: str, session_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        供Agent使用的会话记忆检索：在指定会话中搜索相关历史记录
        返回关键词匹配的最近事件
        """
        query_terms = set(self._tokenize(query))
        if not query_terms:
            return []

        session = self._load_session(session_id)
        scored_events = []
        for event in session["events"]:
            haystack = event["content"].lower()
            hay_terms = set(self._tokenize(haystack))
            hit_terms = query_terms & hay_terms
            if hit_terms:
                score = len(hit_terms) / len(query_terms)
                scored_events.append((score, event))

        scored_events.sort(key=lambda x: (x[0], x[1]["timestamp"]), reverse=True)
        return [e for _, e in scored_events[:top_k]]

    # ===================== 归档清理 =====================

    def _archive_and_trim(self, session_id: str, session: Dict[str, Any]):
        """会话事件超限时归档旧事件，保留最新部分"""
        old_events = session["events"][:-self.max_events_per_session]
        # 归档文件按会话ID+日期命名
        arch_name = f"{session_id}_archive_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        arch_path = os.path.join(self.archive_dir, arch_name)
        archive_data = {
            "id": session_id,
            "archived_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event_count": len(old_events),
            "events": old_events,
        }
        try:
            with open(arch_path, "w", encoding="utf-8") as f:
                json.dump(archive_data, f, ensure_ascii=False, indent=2)
            log_info("会话时序记忆", f"会话[{session_id}]归档{len(old_events)}条旧事件 → {arch_path}")
            # 只保留最新部分
            session["events"] = session["events"][-self.max_events_per_session:]
        except Exception as e:
            log_error("会话时序记忆归档失败", f"会话={session_id}", e)

    def cleanup_expired(self) -> int:
        """清理过期会话日志（超过最大保留天数的会话），返回清理数量"""
        expired_sessions = []
        now = datetime.now()
        for fname in os.listdir(self.storage_dir):
            if not fname.endswith(".json"):
                continue
            session_id = fname[:-5]
            path = self._session_path(session_id)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    session = json.load(f)
                updated_at = datetime.strptime(
                    session.get("updated_at", ""),
                    "%Y-%m-%d %H:%M:%S",
                )
                if (now - updated_at) > timedelta(days=self.max_retention_days):
                    expired_sessions.append((session_id, path))
            except Exception as e:
                log_warn("会话时序记忆清理", f"解析会话{session_id}失败，跳过", e)

        # 删除过期会话
        for session_id, path in expired_sessions:
            try:
                os.remove(path)
                log_info("会话时序记忆", f"清理过期会话[{session_id}]")
            except Exception as e:
                log_error("会话时序记忆清理失败", f"会话={session_id}", e)

        return len(expired_sessions)

    def archive_session(self, session_id: str) -> bool:
        """手动归档指定会话到归档目录"""
        session = self._load_session(session_id)
        arch_name = f"{session_id}_full_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        arch_path = os.path.join(self.archive_dir, arch_name)
        try:
            with open(arch_path, "w", encoding="utf-8") as f:
                json.dump(session, f, ensure_ascii=False, indent=2)
            # 归档后清空当前会话
            os.remove(self._session_path(session_id))
            log_info("会话时序记忆", f"会话[{session_id}]已手动归档 → {arch_path}")
            return True
        except Exception as e:
            log_error("会话时序记忆手动归档失败", f"会话={session_id}", e)
            return False

    # ===================== 统计 =====================

    def count_sessions(self) -> int:
        """当前会话数量"""
        return len([f for f in os.listdir(self.storage_dir) if f.endswith(".json")])

    def count_archives(self) -> int:
        """归档文件数量"""
        return len([f for f in os.listdir(self.archive_dir) if f.endswith(".json")])


# 全局单例
session_temporal_memory = SessionTemporalMemory()