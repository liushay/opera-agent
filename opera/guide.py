# opera/guide.py 个性化学戏路线
# 根据用户需求生成阶梯课程，并按 session 记录进度（对接三层记忆）
import json
import random
from typing import Dict, Any, List

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

import config
from rag.vectorstore import hybrid_retrieve
from utils.logger import log_info, log_warn

DEFAULT_DAYS = 7
MAX_DAYS = 14


def _get_llm():
    return ChatOllama(model=config.LLM_MODEL, temperature=0.3)


def _retrieve_course_material(topic: str, day_topic: str) -> str:
    """检索知识库课程素材"""
    try:
        docs = hybrid_retrieve(f"{topic} {day_topic}")
        if not docs:
            return ""
        return "\n".join([doc.page_content for doc in docs[:3]])
    except Exception as e:
        log_warn("学戏路线", f"知识库检索失败：{e}")
        return ""


def generate_course(
    topic: str,
    days: int,
    session_id: str,
    level: str = "入门",
) -> Dict[str, Any]:
    """
    生成学戏课程路线
    Args:
        topic: 用户想学的主题（如京剧、昆曲、越剧）
        days: 课程天数（默认7，最大14）
        session_id: 会话ID（存储课程进度）
        level: 用户水平（入门/进阶）
    Returns:
        {
            "course_id": 课程ID,
            "topic": 主题,
            "days": 天数,
            "level": 水平,
            "outline": [{"day": 1, "title": "...", "content": "...", "quiz": "..."}],
            "created_at": 创建时间
        }
    """
    log_info("学戏路线", f"生成课程：主题={topic} 天数={days}")
    days = max(1, min(days or DEFAULT_DAYS, MAX_DAYS))

    # 检索整体素材
    material = _retrieve_course_material(topic, "戏曲基础")

    prompt = f"""你是戏曲教育课程设计师。请为{topic}设计一套{days}天的学戏课程路线。

参考素材：
{material or "（知识库暂无该主题素材，请基于戏曲通识设计）"}

用户水平：{level}

要求：
1. 课程循序渐进：Day1 从基础概念引入，中间逐步深入，最后一天做综合回顾
2. 每天包含：标题、学习内容（80-150字）、一个互动问题（考察当天所学）
3. 内容要具体实用，如行当、声腔、代表剧目、名家、经典唱段等

输出严格JSON：
{{
  "outline": [
    {{"day": 1, "title": "当天主题", "content": "当天学习内容", "quiz": "互动问题"}},
    ...
  ]
}}
只输出JSON。"""
    try:
        llm = _get_llm()
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(content[start:end])
            outline = data.get("outline", [])
        else:
            raise json.JSONDecodeError("无JSON", content, 0)
    except Exception as e:
        log_warn("学戏路线", f"课程生成失败，降级为模板：{e}")
        outline = generate_fallback_course(topic, days)

    course_id = f"course_{random.randint(1000, 9999)}"
    import datetime
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 记录课程到任务级记忆 + 会话时序记忆
    try:
        from agent.memory import memory_manager
        memory_manager.record_thinking(
            session_id,
            f"生成学戏课程：{topic}，{days}天，课程ID={course_id}，当前进度=Day0（未开始）"
        )
        memory_manager.record_dialogue(
            session_id, "user", f"我想学习{topic}，请给我制定{days}天学戏路线"
        )
        if course_id:
            try:
                memory_manager.task.create_task(
                    course_id,
                    title=f"{topic}{days}天学戏课程",
                    description=f"用户制定的{days}天{topic}学戏计划",
                    objectives=[f"完成{days}天课程", f"掌握{topic}基础"],
                )
            except Exception:
                pass
    except Exception as e:
        log_warn("学戏路线", f"记录课程失败：{e}")

    return {
        "course_id": course_id,
        "topic": topic,
        "days": days,
        "level": level,
        "outline": outline,
        "created_at": created_at,
    }


def get_course_progress(session_id: str, topic: str = "") -> Dict[str, Any]:
    """
    查询用户当前学戏进度（从会话时序记忆获取最近课程轨迹）
    Returns:
        {"current_day": 当前第几天, "total_days": 总天数, "progress_rate": "30%", "recent_topics": [...]}
    """
    try:
        from agent.memory import memory_manager
        events = memory_manager.session.get_events(session_id, event_type="thinking", limit=50)
        course_events = [
            e for e in events
            if "学戏课程" in e.get("content", "") or "课程进度" in e.get("content", "")
        ]

        # 解析最近一次课程信息
        current_day = 0
        total_days = 0
        recent_topics = []
        for ev in reversed(course_events):
            content = ev.get("content", "")
            import re
            m = re.search(r"进度=Day(\d+)", content)
            if m:
                current_day = int(m.group(1))
            m2 = re.search(r"(\d+)天", content)
            if m2 and not total_days:
                total_days = int(m2.group(1))
            if topic:
                m3 = re.search(rf"课程：{topic}，", content)
            else:
                m3 = re.search(r"课程：(.+?)，(\d+)天", content)
            if m3:
                t = m3.group(1) if m3.lastindex and m3.lastindex >= 1 else topic
                if t and t not in recent_topics:
                    recent_topics.append(t)

        if not total_days:
            total_days = DEFAULT_DAYS
        progress_rate = f"{int(current_day / total_days * 100)}%"
        return {
            "current_day": current_day,
            "total_days": total_days,
            "progress_rate": progress_rate,
            "recent_topics": recent_topics[:5],
        }
    except Exception as e:
        log_warn("学戏路线", f"读取进度失败：{e}")
        return {"current_day": 0, "total_days": DEFAULT_DAYS, "progress_rate": "0%", "recent_topics": []}


def generate_fallback_course(topic: str, days: int) -> List[Dict[str, Any]]:
    """课程生成失败的降级模板"""
    template_days = [
        ("认识{topic}", "了解{topic}的起源、发展脉络与行当分类", "戏曲四大行当是哪些？"),
        ("听一段经典", "欣赏{topic}的经典选段，感受声腔韵味", "你今天听的选段属于哪个行当？"),
        ("认识一位名家", "了解一位{topic}的代表人物及其艺术风格", "这位名家的代表作是什么？"),
        ("学习一个术语", "掌握一个{topic}的专业术语及其含义", "这个术语在表演中起什么作用？"),
        ("剧目赏析", "深入赏析一个经典剧目的剧情与艺术特色", "这个剧目的核心冲突是什么？"),
        ("实践与体验", "尝试跟唱/模仿一个片段，体会表演要领", "你模仿时觉得最难的部分是什么？"),
        ("综合回顾", "总结所学，梳理{topic}知识脉络", "请用一句话介绍你了解的{topic}"),
    ]
    outline = []
    for i in range(1, days + 1):
        if i <= len(template_days):
            title, content, quiz = template_days[i - 1]
            outline.append({
                "day": i,
                "title": title.format(topic=topic),
                "content": content.format(topic=topic),
                "quiz": quiz.format(topic=topic),
            })
        else:
            outline.append({
                "day": i,
                "title": f"深入探索：{topic}专题{i - len(template_days) + 1}",
                "content": f"进一步深入{topic}的某个专题方向，结合所学进行延伸。",
                "quiz": "你学到了什么新的知识点？",
            })
    return outline