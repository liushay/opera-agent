# opera/quiz.py 知识闯关（动态测评）
# 按主题+难度出题 / 判题讲解 / 降级模板题
# 第五部分改造：动态灵活测评（移除固定题量）
#   - 用户可要求任意数量（generate_quiz_batch 动态生成 count 道题）
#   - 支持参考文档约束：reference_required=true 时严格基于文档出题
#   - 支持来源校验：出题后自动校验知识点来源，不匹配自动重生成
import json
import random
import re
from typing import Dict, Any, List, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

import config
from rag.vectorstore import hybrid_retrieve
from utils.logger import log_info, log_warn
from agent.llm_utils import invoke_with_retry

QUIZ_TOPICS = ["行当（生旦净丑）", "经典剧目", "戏曲流派", "戏曲历史", "戏曲术语"]
DIFFICULTY_LEVELS = ["小白", "入门", "票友", "老戏骨"]


def _get_llm():
    return ChatOllama(model=config.LLM_MODEL, temperature=0.4)


def _retrieve_topic_material(topic: str, reference_text: str = "") -> str:
    """按主题检索知识库，作为出题素材；若提供参考文档则优先以参考文档为素材"""
    if reference_text:
        return reference_text[:2000]
    try:
        docs = hybrid_retrieve(topic + " 戏曲知识")
        if not docs:
            return ""
        return "\n".join([doc.page_content for doc in docs[:4]])
    except Exception as e:
        log_warn("知识闯关", f"知识库检索失败：{e}")
        return ""


def _parse_question_json(content: str) -> Dict[str, Any]:
    """容错解析出题JSON"""
    start = content.find("{")
    end = content.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("无有效JSON")
    return json.loads(content[start:end])


def _validate_question_source(question: Dict[str, Any], material: str) -> bool:
    """来源校验：检查题目的知识点是否能在出题素材中找到依据（防止脱离文档）"""
    if not material:
        # 无素材时不强制校验（避免误伤），由参考文档约束开关控制
        return True
    # 取题干+正确答案+讲解拼接，粗略判断与素材是否有共同关键词
    combined = (question.get("question", "") + question.get("answer", "")
                + question.get("explanation", ""))
    # 简单分词：取2字及以上中文词组（粗粒度）
    terms = set(re.findall(r"[\u4e00-\u9fff]{2,6}", combined))
    material_terms = set(re.findall(r"[\u4e00-\u9fff]{2,6}", material))
    hit = terms & material_terms
    # 至少命中2个词组视为有依据
    return len(hit) >= 2


def generate_quiz(topic: str, difficulty: str, session_id: str,
                  exclude_topics: List[str] = None) -> Dict[str, Any]:
    """生成一道选择题（兼容原接口，返回单题）"""
    log_info("知识闯关", f"出题：主题={topic} 难度={difficulty}")
    if difficulty not in DIFFICULTY_LEVELS:
        difficulty = "入门"
    if topic == "混合" or not topic:
        available = [t for t in QUIZ_TOPICS if t not in (exclude_topics or [])]
        topic = random.choice(available or QUIZ_TOPICS)

    material = _retrieve_topic_material(topic)
    if material:
        material = material[:1500]
    if "知识库暂无" not in material and not material:
        material = ""

    difficulty_rule = {
        "小白": "基础常识题，选项中只有一个明显正确的",
        "入门": "常识题，稍有深度",
        "票友": "专业题，需要了解戏曲流派/行当细节",
        "老戏骨": "高阶专业题，冷门知识或细节辨析",
    }[difficulty]
    difficulty_hint = {
        "小白": "选项简单直白，陷阱少",
        "入门": "可设置一个易混淆选项",
        "票友": "可设置两个易混淆选项",
        "老戏骨": "选项精细，考核细节",
    }[difficulty]

    prompt = f"""你是戏曲知识出题官。请根据以下素材出一道{difficulty}难度的{topic}知识选择题。

素材（供出题参考）：
{material or "（知识库暂无该主题素材，请基于戏曲常识出题）"}

【难度要求】{difficulty_rule}
【选项要求】{difficulty_hint}

输出严格JSON：
{{
  "question": "题干",
  "options": {{"A": "选项A", "B": "选项B", "C": "选项C", "D": "选项D"}},
  "answer": "正确选项字母 A/B/C/D",
  "explanation": "60-100字知识点讲解",
  "tips": "10-20字提示"
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
        else:
            raise json.JSONDecodeError("无JSON", content, 0)
    except Exception as e:
        log_warn("知识闯关", f"出题LLM解析失败，降级为模板题：{e}")
        return generate_fallback_question(topic, difficulty)

    try:
        from agent.memory import memory_manager
        memory_manager.record_thinking(
            session_id, f"知识闯关出题：{topic}（{difficulty}）→ {data.get('question', '')[:50]}"
        )
    except Exception as e:
        log_warn("知识闯关", f"记录出题轨迹失败：{e}")

    return {
        "question_id": f"quiz_{random.randint(1000, 9999)}",
        "topic": topic,
        "difficulty": difficulty,
        "question": data.get("question", ""),
        "options": data.get("options", {}),
        "answer": data.get("answer", "A"),
        "explanation": data.get("explanation", ""),
        "tips": data.get("tips", ""),
    }


def generate_quiz_batch(
    topic: str,
    difficulty: str,
    session_id: str,
    count: int = 1,
    reference_text: str = "",
    reference_required: bool = False,
) -> Dict[str, Any]:
    """
    动态生成任意数量测评题（第四部分：动态灵活测评）。
    Args:
        topic: 主题
        difficulty: 难度
        session_id: 会话ID
        count: 题目数量（完全跟随用户需求，如"要4道题"→count=4）
        reference_text: 参考文档文本（用户要求严格基于文档时传入）
        reference_required: 是否强制基于参考文档
    Returns:
        {"questions": [题目...], "count": N, "topic": ..., "difficulty": ...}
    """
    count = max(1, int(count or 1))
    log_info("知识闯关", f"动态出题：{count}道，主题={topic} 难度={difficulty}")

    if difficulty not in DIFFICULTY_LEVELS:
        difficulty = "入门"
    if topic == "混合" or not topic:
        topic = random.choice(QUIZ_TOPICS)

    # 素材：优先参考文档，否则检索知识库
    material = _retrieve_topic_material(topic, reference_text=reference_text)
    if material:
        material = material[:2000]
    if not material:
        material = ""

    # 文档约束提示
    ref_rule = ""
    if reference_required:
        ref_rule = (
            "\n【强制约束】你必须严格且完全基于上面提供的参考文档出题，"
            "禁止使用模型预训练知识、禁止外部知识点、禁止脑补。"
            "题干、选项、答案、讲解中的知识点都必须能在文档中找到依据。"
        )

    prompt = f"""你是戏曲知识出题官。请根据以下素材生成{count}道{difficulty}难度的{topic}知识选择题。
要求：{count}道题必须各不相同、覆盖该主题的不同知识点。完全跟随用户题目数量需求，不多不少正好{count}道。

素材（唯一出题依据）：
{material or "（知识库暂无该主题素材，请基于戏曲常识出题）"}
{ref_rule}

【难度要求】
{"基础常识题，选项中只有一个明显正确的" if difficulty=="小白" else "常识题，稍有深度" if difficulty=="入门" else "专业题，需要了解戏曲流派/行当细节" if difficulty=="票友" else "高阶专业题，冷门知识或细节辨析"}

输出严格JSON数组（不要输出多余文字）：
[
  {{
    "question": "题干1",
    "options": {{"A": "选项A", "B": "选项B", "C": "选项C", "D": "选项D"}},
    "answer": "正确选项字母 A/B/C/D",
    "explanation": "60-100字知识点讲解（必须基于素材）",
    "tips": "10-20字提示"
  }},
  ...
]
一共{count}个对象。只输出JSON数组。"""

    questions = []
    # 最多尝试3次整体生成（含来源校验失败重生成），避免无限循环
    for _attempt in range(3):
        content = invoke_with_retry(
            [prompt],
            temperature=0.4,
            task_name=f"动态出题({count}道)",
        )
        # 尝试解析为数组；若解析成单个对象则包装为数组
        data = None
        try:
            parsed = json.loads(content.strip())
            if isinstance(parsed, list):
                data = parsed
            elif isinstance(parsed, dict):
                data = [parsed]
        except Exception:
            # 提取JSON数组（容错）
            import re as _re
            m = _re.search(r"\[.*\]", content, _re.S)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                    data = parsed if isinstance(parsed, list) else [parsed]
                except Exception:
                    data = None
        if not data:
            log_warn("知识闯关", f"第{_attempt+1}次解析失败，重试...")
            continue

        # 来源校验：逐题校验知识点是否基于素材
        if reference_required and material:
            valid_qs = []
            for q in data:
                q = q if isinstance(q, dict) else {}
                if _validate_question_source(q, material):
                    valid_qs.append(q)
            if len(valid_qs) < count:
                log_warn("知识闯关", f"来源校验未通过：仅{len(valid_qs)}/{count}道有依据，重试...")
                data = valid_qs if valid_qs else None
                if data:
                    questions = _finalize_questions(data, topic, difficulty, session_id)
                continue
        questions = _finalize_questions(data, topic, difficulty, session_id)
        if questions:
            break

    # 若始终失败，用模板题兜底（保证不空）
    if not questions:
        log_warn("知识闯关", "动态出题失败，降级为模板题兜底")
        questions = [generate_fallback_question(topic, difficulty)]

    return {
        "questions": questions,
        "count": len(questions),
        "topic": topic,
        "difficulty": difficulty,
        "dynamic": True,
    }


def _finalize_questions(data: List[Dict], topic: str, difficulty: str, session_id: str) -> List[Dict]:
    """将解析出的原始题目格式化为标准结构"""
    questions = []
    for q in data:
        if not isinstance(q, dict):
            continue
        questions.append({
            "question_id": f"quiz_{random.randint(10000, 99999)}",
            "topic": topic,
            "difficulty": difficulty,
            "question": q.get("question", ""),
            "options": q.get("options", {}),
            "answer": q.get("answer", "A"),
            "explanation": q.get("explanation", ""),
            "tips": q.get("tips", ""),
        })
    return questions


def check_answer(question_id: str, user_answer: str, correct_answer: str,
                 explanation: str, session_id: str) -> Dict[str, Any]:
    """判题并返回讲解"""
    is_correct = user_answer.strip().upper() == correct_answer.strip().upper()
    log_info("知识闯关", f"判题：{question_id} 用户答{'对' if is_correct else '错'}")
    encouragement = (
        "答对啦！看来你已经掌握这一关，继续挑战下一关吧！🎉"
        if is_correct else
        "答错了没关系，看看下面的讲解，下次一定行！💪"
    )
    try:
        from agent.memory import memory_manager
        memory_manager.record_thinking(
            session_id,
            f"知识闯关判题：{question_id} {'正确' if is_correct else '错误'}，讲解：{explanation[:50]}"
        )
    except Exception as e:
        log_warn("知识闯关", f"记录判题失败：{e}")
    return {
        "correct": is_correct,
        "user_answer": user_answer,
        "correct_answer": correct_answer,
        "explanation": explanation,
        "encouragement": encouragement,
    }


def generate_fallback_question(topic: str, difficulty: str) -> Dict[str, Any]:
    """出题失败时的降级模板题"""
    fallback_bank = [
        {
            "topic": "行当（生旦净丑）",
            "question": "戏曲中'净'行通常指哪类角色？",
            "options": {"A": "老生", "B": "花脸", "C": "青衣", "D": "小生"},
            "answer": "B",
            "explanation": "'净'行即花脸，以面部化妆（脸谱）为标志，如包拯、曹操等角色。",
            "tips": "想想哪些角色画了脸谱",
        },
        {
            "topic": "经典剧目",
            "question": "《牡丹亭》的作者是谁？",
            "options": {"A": "关汉卿", "B": "王实甫", "C": "汤显祖", "D": "孔尚任"},
            "answer": "C",
            "explanation": "《牡丹亭》是明代汤显祖的代表作，讲述杜丽娘与柳梦梅的爱情故事。",
            "tips": "明代最著名的戏曲家",
        },
        {
            "topic": "戏曲流派",
            "question": "京剧'梅派'的创始人是？",
            "options": {"A": "程砚秋", "B": "尚小云", "C": "荀慧生", "D": "梅兰芳"},
            "answer": "D",
            "explanation": "梅派由梅兰芳创立，以雍容华贵、歌舞并重著称，是四大名旦之首。",
            "tips": "四大名旦之首",
        },
    ]
    for q in fallback_bank:
        if q["topic"] == topic:
            return {
                "question_id": f"quiz_fallback_{random.randint(1000, 9999)}",
                "topic": topic, "difficulty": difficulty,
                "question": q["question"], "options": q["options"],
                "answer": q["answer"], "explanation": q["explanation"],
                "tips": q["tips"],
            }
    return {
        "question_id": f"quiz_fallback_{random.randint(1000, 9999)}",
        "topic": topic, "difficulty": difficulty,
        "question": f"关于『{topic}』，下面哪项描述是正确的？",
        "options": {"A": "选项A", "B": "选项B", "C": "选项C", "D": "选项D"},
        "answer": "A",
        "explanation": "本题为降级模板题，请稍后再试出题。",
        "tips": "请重试",
    }