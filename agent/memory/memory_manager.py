# agent/memory/memory_manager.py 统一记忆管理器
# 功能：
#   1. 统一管理三层记忆：永久静态 / 任务级 / 会话时序
#   2. 为上层Agent提供统一检索与记录接口
#   3. 保持各记忆层独立存储、完全隔离
import json
from typing import List, Dict, Any, Optional

from agent.memory.permanent_memory import PermanentMemory, permanent_memory
from agent.memory.task_memory import TaskMemory, task_memory
from agent.memory.session_memory import SessionTemporalMemory, session_temporal_memory
from utils.logger import log_info, log_warn, log_error

from utils.rag_exceptions import BaseRAGException


class MemoryManagerException(BaseRAGException):
    """统一记忆管理器异常"""
    code = 5100
    msg = "统一记忆管理器操作失败"


class MemoryManager:
    """
    统一记忆管理器：协调三层隔离记忆
    - permanent: 永久静态记忆（全局共享）
    - task:      任务级记忆（任务隔离）
    - session:   会话时序记忆（时序轨迹）
    """

    def __init__(
        self,
        permanent: PermanentMemory = None,
        task: TaskMemory = None,
        session: SessionTemporalMemory = None,
    ):
        self.permanent = permanent or permanent_memory
        self.task = task or task_memory
        self.session = session or session_temporal_memory

    # ===================== 统一检索接口 =====================

    def retrieve_for_agent(
        self,
        query: str,
        session_id: str = "",
        task_id: Optional[str] = None,
        include_permanent: bool = True,
        include_task: bool = True,
        include_session: bool = True,
        top_k: int = 3,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        统一检索三层记忆（供Agent调用）
        返回结构：
        {
          "permanent": [...],   # 永久静态记忆检索结果
          "task": [...],        # 任务级记忆检索结果
          "session": [...]      # 会话时序记忆检索结果
        }
        """
        result = {"permanent": [], "task": [], "session": []}

        # 1. 永久静态记忆（全局共享）
        if include_permanent:
            result["permanent"] = self.permanent.search_for_agent(query, top_k=top_k)

        # 2. 任务级记忆（指定任务）
        if include_task:
            if task_id:
                # 如果指定了task_id，直接获取该任务的摘要
                try:
                    t = self.task.get_task(task_id)
                    result["task"].append({
                        "id": t["id"],
                        "title": t.get("title", ""),
                        "description": t.get("description", ""),
                        "status": t.get("status", ""),
                        "objectives": t.get("objectives", [])[:3],
                        "constraints": t.get("constraints", [])[:3],
                        "intermediate_results": t.get("intermediate_results", [])[-3:],
                    })
                except Exception as e:
                    log_warn("统一记忆管理器", f"获取任务{task_id}失败：{e}")
            else:
                # 未指定task_id，按关键词搜索全部任务
                result["task"] = self.task.search_for_agent(query, top_k=top_k)

        # 3. 会话时序记忆（指定会话）
        if include_session and session_id:
            result["session"] = self.session.search_for_agent(
                query, session_id=session_id, top_k=top_k
            )

        return result

    def format_memory_context(
        self,
        query: str,
        session_id: str = "",
        task_id: Optional[str] = None,
    ) -> str:
        """
        生成格式化记忆上下文文本（供LLM提示词使用）
        将三层记忆检索结果拼接为可读文本
        """
        mem_data = self.retrieve_for_agent(
            query=query,
            session_id=session_id,
            task_id=task_id,
        )

        parts = []

        # 永久静态记忆
        if mem_data["permanent"]:
            perm_lines = ["【永久静态记忆】"]
            for item in mem_data["permanent"]:
                title = item.get("title", "")
                content = item.get("content", "")
                if title:
                    perm_lines.append(f"- {title}：{content}")
                else:
                    perm_lines.append(f"- {content}")
            parts.append("\n".join(perm_lines))

        # 任务级记忆
        if mem_data["task"]:
            task_lines = ["【任务级记忆】"]
            for item in mem_data["task"]:
                title = item.get("title", "")
                desc = item.get("description", "")
                status = item.get("status", "")
                objs = item.get("objectives", [])
                task_lines.append(f"- 任务[{item.get('id', '')}] 状态:{status}")
                if title:
                    task_lines.append(f"  标题：{title}")
                if desc:
                    task_lines.append(f"  描述：{desc}")
                if objs:
                    task_lines.append(f"  目标：{'；'.join(objs[:3])}")
            parts.append("\n".join(task_lines))

        # 会话时序记忆
        if mem_data["session"]:
            sess_lines = ["【会话时序记忆】"]
            for event in mem_data["session"]:
                etype = event.get("event_type", "")
                content = event.get("content", "")
                ts = event.get("timestamp", "")
                sess_lines.append(f"- [{ts}] ({etype}) {content[:120]}")
            parts.append("\n".join(sess_lines))

        return "\n\n".join(parts)

    # ===================== 统一记录接口 =====================

    def record_dialogue(self, session_id: str, role: str, content: str):
        """记录对话轨迹"""
        self.session.add_dialogue(session_id, role, content)

    def record_thinking(self, session_id: str, content: str):
        """记录思考过程"""
        self.session.add_thinking(session_id, content)

    def record_tool_call(self, session_id: str, tool_name: str, params: str):
        """记录工具调用"""
        self.session.add_tool_call(session_id, tool_name, params)

    def record_tool_result(self, session_id: str, tool_name: str, result: str):
        """记录工具结果"""
        self.session.add_tool_result(session_id, tool_name, result)

    def record_error(self, session_id: str, error_msg: str, stack: str = ""):
        """记录报错"""
        self.session.add_error(session_id, error_msg, stack)

    def record_modify(self, session_id: str, description: str, target: str = ""):
        """记录修改"""
        self.session.add_modify(session_id, description, target)

    # ===================== 统计信息 =====================

    def get_stats(self) -> Dict[str, Any]:
        """获取三层记忆的统计信息"""
        return {
            "permanent_memory": {
                "total_count": self.permanent.count(),
                "categories": {
                    cat: len(items) for cat, items in self.permanent.get_all().items()
                },
                "storage": "本地JSON文件（全局共享）",
            },
            "task_memory": {
                "total_tasks": self.task.count(),
                "storage": "本地JSON目录（按任务隔离）",
            },
            "session_memory": {
                "active_sessions": self.session.count_sessions(),
                "archives": self.session.count_archives(),
                "storage": "本地JSON目录（按会话隔离）",
            },
        }


# 全局单例
memory_manager = MemoryManager()