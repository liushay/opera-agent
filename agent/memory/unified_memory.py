# agent/memory/unified_memory.py 统一认知记忆抽象（Working/Episodic/Semantic）
# 功能：
#   1. 将现有三层 JSON Memory 映射为认知科学标准的三类记忆
#      - Working Memory（工作记忆）= 会话时序记忆（短期当前上下文）
#      - Episodic Memory（情景记忆）= 任务级记忆（经历/任务轨迹）
#      - Semantic Memory（语义记忆）= 永久静态记忆（知识/规则）
#   2. 保持旧接口完全兼容（permanent_memory/task_memory/session_temporal_memory 不变）
#   3. 提供统一的 retrieve/record/cleanup 接口供 Agent 与 MCP 使用
from typing import List, Dict, Any, Optional

from agent.memory.permanent_memory import PermanentMemory, permanent_memory
from agent.memory.task_memory import TaskMemory, task_memory
from agent.memory.session_memory import SessionTemporalMemory, session_temporal_memory
from utils.logger import log_info, log_warn


class WorkingMemory:
    """工作记忆：短期会话上下文（映射到会话时序记忆）"""

    def __init__(self, store: SessionTemporalMemory):
        self.store = store

    def add(self, session_id: str, event_type: str, content: str, metadata: dict = None) -> str:
        return self.store.add_event(session_id, event_type, content, metadata)

    def get(self, session_id: str, limit: int = 50) -> List[Dict]:
        return self.store.get_events(session_id, limit=limit)

    def clear(self, session_id: str) -> bool:
        """清理指定会话的工作记忆（归档）"""
        return self.store.archive_session(session_id)

    def summarize(self, session_id: str) -> Dict:
        """获取工作记忆概要"""
        return self.store.get_session_info(session_id)


class EpisodicMemory:
    """情景记忆：任务经历（映射到任务级记忆）"""

    def __init__(self, store: TaskMemory):
        self.store = store

    def remember(self, task_id: str, **kwargs) -> Dict:
        """记录一个任务经历"""
        task = self.store.get_task(task_id)
        # 将调用参数作为情景写入 intermediate_results
        result_id = self.store.add_intermediate_result(
            task_id, {"type": "episode", "content": str(kwargs)}
        )
        return {"task_id": task_id, "episode_id": result_id}

    def recall(self, query: str, top_k: int = 3) -> List[Dict]:
        """按关键词回忆相关任务经历"""
        return self.store.search_for_agent(query, top_k=top_k)

    def create_episode(self, **kwargs) -> Dict:
        """创建一个新的任务经历空间"""
        return self.store.create_task(**kwargs)


class SemanticMemory:
    """语义记忆：知识/规则/偏好（映射到永久静态记忆）"""

    def __init__(self, store: PermanentMemory):
        self.store = store

    def declare(self, content: str, category: str = "domain_spec", **kwargs) -> str:
        """声明一条语义知识"""
        return self.store.add(category=category, content=content, **kwargs)

    def query(self, query: str, top_k: int = 5) -> List[Dict]:
        """语义检索"""
        return self.store.search_for_agent(query, top_k=top_k)

    def forget(self, mem_id: str) -> bool:
        """遗忘一条语义知识"""
        return self.store.delete(mem_id)


class UnifiedCognitiveMemory:
    """
    统一认知记忆管理器
    将三类记忆统一为一个认知视角：
      working:  WorkingMemory  → session_temporal_memory
      episodic: EpisodicMemory → task_memory
      semantic: SemanticMemory → permanent_memory
    """

    def __init__(
        self,
        permanent: PermanentMemory = None,
        task: TaskMemory = None,
        session: SessionTemporalMemory = None,
    ):
        p = permanent or permanent_memory
        t = task or task_memory
        s = session or session_temporal_memory

        self.semantic = SemanticMemory(p)
        self.episodic = EpisodicMemory(t)
        self.working = WorkingMemory(s)

        # 保留底层引用（兼容旧代码）
        self.permanent_mean = p

    def retrieve_all(
        self,
        query: str,
        session_id: str = "",
        task_id: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict[str, List[Dict]]:
        """统一检索三类记忆"""
        result = {}
        # Semantic
        result["semantic"] = self.semantic.query(query, top_k=top_k)
        # Episodic
        if task_id:
            try:
                t = self.episodic.store.get_task(task_id)
                result["episodic"] = [{
                    "id": t["id"],
                    "title": t.get("title", ""),
                    "status": t.get("status", ""),
                    "objectives": t.get("objectives", [])[:3],
                    "intermediate_results": t.get("intermediate_results", [])[-3:],
                }]
            except Exception:
                result["episodic"] = []
        else:
            result["episodic"] = self.episodic.recall(query, top_k=top_k)
        # Working
        result["working"] = (
            self.working.get(session_id, limit=top_k * 5) if session_id else []
        )
        return result

    def format_context(
        self,
        query: str,
        session_id: str = "",
        task_id: Optional[str] = None,
    ) -> str:
        """生成 LLM 可读的分层记忆上下文"""
        data = self.retrieve_all(query, session_id=session_id, task_id=task_id)
        parts = []

        if data["semantic"]:
            lines = ["【语义记忆（知识/规则）】"]
            for item in data["semantic"]:
                title = item.get("title", "")
                content = item.get("content", "")
                lines.append(f"- {title}：{content}" if title else f"- {content}")
            parts.append("\n".join(lines))

        if data["episodic"]:
            lines = ["【情景记忆（任务经历）】"]
            for item in data["episodic"]:
                lines.append(
                    f"- 任务[{item.get('id', '')}] 状态:{item.get('status', '')} "
                    f"目标:{'；'.join(item.get('objectives', []))}"
                )
            parts.append("\n".join(lines))

        if data["working"]:
            lines = ["【工作记忆（对话上下文）】"]
            for event in data["working"][-10:]:
                ts = event.get("timestamp", "")
                etype = event.get("event_type", "")
                content = event.get("content", "")
                lines.append(f"- [{ts}] ({etype}) {content[:100]}")
            parts.append("\n".join(lines))

        return "\n\n".join(parts)


# 全局单例（统一认知记忆管理器，保持旧接口兼容）
unified_memory = UnifiedCognitiveMemory()