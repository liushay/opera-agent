# agent/memory/__init__.py 分层隔离记忆模块
# 功能：
#   1. 永久静态记忆：系统规则、领域规范、技术文档、用户固定偏好，全局共享
#   2. 任务级记忆：每个任务独立记忆空间，支持父子任务嵌套
#   3. 会话时序记忆：对话轨迹、思考、工具调用、结果、报错、修改记录，自动归档清理
#   4. 统一认知记忆抽象：Working/Episodic/Semantic（兼容旧接口）
from agent.memory.permanent_memory import PermanentMemory, permanent_memory
from agent.memory.task_memory import TaskMemory, task_memory
from agent.memory.session_memory import SessionTemporalMemory, session_temporal_memory
from agent.memory.memory_manager import MemoryManager, memory_manager
from agent.memory.unified_memory import (
    WorkingMemory,
    EpisodicMemory,
    SemanticMemory,
    UnifiedCognitiveMemory,
    unified_memory,
)

__all__ = [
    "PermanentMemory",
    "permanent_memory",
    "TaskMemory",
    "task_memory",
    "SessionTemporalMemory",
    "session_temporal_memory",
    "MemoryManager",
    "memory_manager",
    # 统一认知记忆
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "UnifiedCognitiveMemory",
    "unified_memory",
]
