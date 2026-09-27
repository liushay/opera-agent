# agent/task_context.py 任务隔离上下文分配器
# 功能：
#   1. 每一次用户新提问 = 全新 task_id（UUID），进入全新 Episodic 任务记忆空间
#   2. 旧 task 的双功能输出、闯关内容、文献内容完全隔离，绝不跨任务残留
#   3. 新提问只会读取本次 task 的内容，彻底杜绝"上次做了A+B，这次只想要A却返回A+B"
# 设计原则：
#   - 不改变三层记忆文件结构（仍是 memory_store/tasks/{task_id}.json）
#   - 只在调用方创建新 task_id，并保证检索时只读取本次 task_id
#   - Working 会话记忆仍可提供聊天连贯文本（对话轨迹），但绝不注入旧任务功能输出
import uuid
from typing import Any, Dict, Optional

from agent.memory import memory_manager, task_memory
from utils.logger import log_info, log_warn


def create_new_task(
    user_query: str,
    title: str = "",
    session_id: str = "",
) -> Dict[str, Any]:
    """
    为每一次用户新提问创建全新任务记忆空间（Episodic）。
    - 每次调用生成新 task_id，不同任务物理隔离
    - 将用户本次核心需求写入任务 requirements，便于本次生成参考
    - 返回任务数据（含 task_id）
    """
    task_id = str(uuid.uuid4())
    try:
        task_data = task_memory.create_task(
            task_id=task_id,
            title=title or f"用户新请求：{user_query[:30]}",
            description="单次独立任务，用于隔离不同轮次请求，彻底解决记忆污染",
            requirements=user_query,
            metadata={"session_id": session_id, "source": "intent_orchestrator"},
        )
        log_info("任务隔离", f"为用户新请求创建任务 [{task_id}]：{user_query[:40]}")
        return task_data
    except Exception as e:
        log_warn("任务隔离", f"创建任务失败，降级为无任务ID：{e}")
        return {"id": task_id, "error": str(e)}


def get_task_context_text(task_id: Optional[str], max_results: int = 3) -> str:
    """
    获取"仅本次任务"的上下文文本（Episodic）。
    - 只读取指定 task_id 的内容，绝不搜索全部历史任务
    - 保证旧任务的双功能输出/闯关/文献内容不会混入本次
    """
    if not task_id:
        return ""
    try:
        t = task_memory.get_task(task_id)
    except Exception as e:
        log_warn("任务隔离", f"读取任务{task_id}失败：{e}")
        return ""
    parts = []
    if t.get("title"):
        parts.append(f"任务标题：{t['title']}")
    if t.get("requirements"):
        parts.append(f"本次任务需求：{t['requirements']}")
    objs = t.get("objectives") or []
    if objs:
        parts.append(f"本次目标：{'；'.join(str(o) for o in objs[:max_results])}")
    intermediate = t.get("intermediate_results") or []
    if intermediate:
        parts.append("本次任务中间产物：")
        for item in intermediate[-max_results:]:
            content = item.get("content", "") if isinstance(item, dict) else str(item)
            parts.append(f"  - {str(content)[:200]}")
    return "\n".join(parts)


def get_current_task_only(
    task_id: Optional[str],
    session_id: str = "",
) -> str:
    """
    【核心记忆隔离函数】
    生成供 LLM 提示词使用的记忆上下文（替代原先跨任务搜索的 format_memory_context）：
      1. Semantic 永久记忆：只微调语气/人设（不参与当前生成逻辑、不提供答题素材）
      2. Episodic 任务记忆：仅本次 task_id 的上下文（不跨任务）
      3. Working 会话记忆：仅最近对话文本（用于聊天连贯，不继承旧任务功能输出）
    关键：绝不含旧任务生成结果（文献/闯关/问答产物）。
    """
    parts = []

    # 1. Semantic 永久记忆（只读人设/偏好，参与微调语气，绝不提供答题素材）
    try:
        perm = memory_manager.permanent.search_for_agent("用户偏好 人设 固定规则", top_k=2)
        perm_lines = []
        for item in perm:
            title = item.get("title", "")
            content = item.get("content", "")
            perm_lines.append(f"- {title}：{content}" if title else f"- {content}")
        if perm_lines:
            parts.append("【长期语义记忆（仅用于人设/语气微调，不参与生成内容）】\n" + "\n".join(perm_lines))
    except Exception as e:
        log_warn("记忆隔离", f"读取永久记忆失败：{e}")

    # 2. Episodic 任务记忆（仅本次任务，绝不跨任务）
    task_text = get_task_context_text(task_id)
    if task_text:
        parts.append("【本次任务记忆（仅当前任务，完全隔离旧任务）】\n" + task_text)

    # 3. Working 会话记忆（仅最近对话文本，用于聊天连贯）
    #    问题3修复：只保留最近"用户"消息，过滤掉旧AI输出（可能含旧任务闯关题目/文献/检索结果）
    if session_id:
        try:
            recent = memory_manager.session.get_recent_dialogue(session_id, limit=6)
            if recent:
                lines = []
                for ev in recent:
                    role = (ev.get("metadata") or {}).get("role", "")
                    content = ev.get("content", "")
                    # 只放行用户消息；AI 历史输出属于旧任务生成结果，绝不注入本次上下文
                    if role != "user":
                        continue
                    lines.append(f"- 用户之前问过：{str(content)[:200]}")
                if lines:
                    parts.append("【最近用户提问（仅用于对话连贯，绝不作为本次生成内容依据）】\n" + "\n".join(lines))
        except Exception as e:
            log_warn("记忆隔离", f"读取会话记忆失败：{e}")

    return "\n\n".join(parts)
