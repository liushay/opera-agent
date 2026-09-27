# evaluation/dynamic_questions.py 动态测评问题集生成器
# 功能（问题2改造）：
#   1. 不再依赖 config.py 中硬编码的固定测评问题集
#   2. 依据传入主题(topic) + 参考文档知识库内容(reference_text / 知识库检索) 自动生成测评问题集合
#   3. 支持生成不同类型问题：事实问答 / 理解类 / 抽取类，覆盖全方面测评
#   4. 返回结构与原 EVAL_QUESTION_SET 完全一致：[{"query": "...", "expected_keywords": [...]}]
#      保证测评指标计算逻辑（Recall/HitRate/MRR/NDCG）无需改动
# 设计原则：
#   - 完全解耦，不改变原有评测计算逻辑，只新增问题来源
#   - LLM 生成失败时降级返回基于知识库文档关键词的规则问题集，保证评测不阻塞
import json
import re
from typing import Any, Dict, List, Optional

import config
from agent.llm_utils import invoke_with_retry, safe_llm_call
from rag.vectorstore import hybrid_retrieve
from utils.logger import log_info, log_warn, log_error


# 测评问题类型（全方面覆盖）
QUESTION_TYPES = [
    ("fact", "事实问答：基于文档中的明确事实提问（如'XXX是什么/谁/何时'）"),
    ("understanding", "理解类：需要综合文档内容理解后回答（如'XXX的原因/意义/作用'）"),
    ("extraction", "抽取类：从文档中抽取关键信息（如'列出/找出文档中提到的XXX'）"),
]


def _build_prompt(
    topic: str,
    reference_text: str,
    question_count: int,
) -> str:
    """构建动态测评问题生成提示词"""
    types_desc = "\n".join(f"  - {t}: {desc}" for t, desc in QUESTION_TYPES)
    return f"""
你是戏曲领域知识库的【测评问题生成专家】。请根据给定主题和参考文档内容，自动生成一批高质量的检索测评问题。

【测评主题】
{topic}

【参考文档内容（知识库检索结果）】
{reference_text[:4000]}

【生成要求】
1. 生成 {question_count} 个测评问题，覆盖以下类型（每种类型至少1个）：
{types_desc}

2. 每个问题必须：
   - 与主题 {topic} 紧密相关
   - 能从参考文档中找到答案依据（问题可基于文档事实回答）
   - 期望关键词须来自于参考文档中的关键实体/术语/专有名词（用于判定检索相关性）
   - 期望关键词数量 3~5 个，必须是文档中实际出现的关键词

3. 输出严格JSON数组，格式：
[
  {{
    "query": "测评问题文本",
    "question_type": "fact|understanding|extraction",
    "expected_keywords": ["关键词1", "关键词2", "关键词3"]
  }}
]
只输出JSON数组，不要输出任何多余文字。
"""


def _extract_keywords_from_docs(docs: List[Any], max_keywords: int = 5) -> List[str]:
    """
    从检索文档中提取高频关键词（规则降级方案）：
    1. 按标点/空白切分为候选词
    2. 长片段继续按「代表/如/是/的/和/与」等连接词二次拆词，得到更精确的术语关键词
    3. 过滤无意义词
    4. 保留长度 2~8 的中文词/术语，按频次排序取前 max_keywords
    """
    stopwords = {
        "的", "了", "是", "在", "和", "与", "及", "或", "对", "为", "从",
        "一个", "一种", "这个", "那个", "可以", "进行", "以及", "戏曲",
        "中国", "就是", "不是", "什么", "如何", "哪些", "为什么", "具有",
        "一种", "以及", "代表", "表示", "象征", "非常", "成为", "属于",
    }
    # 二次拆词连接词：将长片段拆成小术语（去重）
    split_connectors = ("代表", "表示", "是", "如", "的", "和", "与", "及", "或")

    def _split_long_seg(seg: str) -> List[str]:
        """将长片段按连接词拆成更小的候选词"""
        parts = [seg]
        for conn in split_connectors:
            new_parts = []
            for p in parts:
                if conn in p and len(p) > 4:
                    new_parts.extend(p.split(conn))
                else:
                    new_parts.append(p)
            parts = new_parts
        cleaned = []
        for p in parts:
            p = p.strip()
            # 去掉"如关羽"这类以"如"开头的指代短语，保留"关羽"
            p = re.sub(r"^如+", "", p).strip()
            if len(p) >= 2:
                cleaned.append(p)
        return cleaned

    counter: Dict[str, int] = {}
    for doc in docs:
        text = doc.page_content if hasattr(doc, "page_content") else str(doc)
        # 按非中文字符分割，保留中文词组
        for seg in re.split(r"[^\u4e00-\u9fffA-Za-z0-9]+", text):
            seg = seg.strip()
            if not (2 <= len(seg) <= 12):
                continue
            if seg in stopwords:
                continue
            # 长片段二次拆词，得到精确术语（如"红色脸谱代表忠义勇敢"→"红色脸谱"/"忠义勇敢"）
            for sub in _split_long_seg(seg):
                sub = sub.strip()
                if not (2 <= len(sub) <= 8):
                    continue
                if sub in stopwords:
                    continue
                counter[sub] = counter.get(sub, 0) + 1
    # 按出现频次降序，取前 max_keywords
    sorted_kws = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in sorted_kws[:max_keywords]]


def _fallback_question_set(
    topic: str,
    reference_text: str,
    question_count: int,
) -> List[Dict[str, Any]]:
    """
    规则降级方案：LLM 生成失败时，基于知识库检索文档自动生成问题集。
    保证评测流程不因 LLM 异常而中断。
    """
    try:
        docs = hybrid_retrieve(topic)
    except Exception as e:
        log_warn("动态测评降级", f"知识库检索失败：{e}，将使用参考文本作为文档来源")
        docs = []
    if not docs and reference_text:
        # 参考文本直接作为文档
        from langchain_core.documents import Document
        docs = [Document(page_content=reference_text, metadata={})]

    if not docs:
        log_warn("动态测评", "知识库无检索结果且无参考文档，降级生成默认问题")
        return [
            {
                "query": topic,
                "question_type": "fact",
                "expected_keywords": [topic, "戏曲", "艺术", "表演", "文化"],
            }
        ]

    keywords = _extract_keywords_from_docs(docs, max_keywords=5)
    if len(keywords) < 3:
        keywords = (keywords + ["戏曲", "戏剧", "表演", "文化", "艺术"])[:5]

    # 构造不同类型问题
    question_set = []
    topic_short = topic[:20]
    type_templates = [
        ("fact", f"{topic_short}的主要内容是什么？"),
        ("understanding", f"{topic_short}具有哪些重要特点与艺术价值？"),
        ("extraction", f"请从文档中提取关于{topic_short}的关键知识点。"),
    ]
    for i in range(question_count):
        q_type, template = type_templates[i % len(type_templates)]
        question_set.append({
            "query": template,
            "question_type": q_type,
            "expected_keywords": list(keywords),
        })
    return question_set


def generate_dynamic_question_set(
    topic: str,
    reference_text: str = "",
    question_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    动态生成测评问题集（替代原 config.EVAL_QUESTION_SET 硬编码）。
    Args:
        topic: 测评主题（必传）
        reference_text: 参考文档内容（可选，若为空则自动从知识库检索 topic 相关文档）
        question_count: 生成问题数量（默认 config.EVAL_DYNAMIC_QUESTION_COUNT）
    Returns:
        [{"query": str, "question_type": str, "expected_keywords": [str]}]
    """
    count = question_count or getattr(config, "EVAL_DYNAMIC_QUESTION_COUNT", 6)
    # 确保参考文档非空：无参考文本时自动检索知识库
    effective_ref = reference_text
    if not effective_ref:
        try:
            docs = hybrid_retrieve(topic)
            effective_ref = "\n".join(
                d.page_content for d in docs
            )
            log_info("动态测评", f"已自动从知识库检索 {len(docs)} 篇文档作为参考材料")
        except Exception as e:
            log_warn("动态测评", f"知识库检索参考文档失败：{e}")
            effective_ref = ""

    log_info("动态测评", f"开始为主题[{topic}]动态生成{count}道测评问题")
    prompt = _build_prompt(topic, effective_ref, count)

    try:
        # 问题生成 LLM 调用：缩短超时 + 限制重试，避免测评接口整体超时
        # （LLM 失败立即降级为规则生成，绝不阻塞）
        content = invoke_with_retry(
            [prompt],
            temperature=0.3,
            timeout=getattr(config, "EVAL_QUESTION_GEN_TIMEOUT", 45),
            max_retry=0,
            task_name="动态测评问题生成",
        )
        start = content.find("[")
        end = content.rfind("]") + 1
        if start < 0 or end <= start:
            raise ValueError("无有效JSON数组")
        data = json.loads(content[start:end])
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("生成结果非数组或为空")
        question_set = []
        for item in data:
            query = str(item.get("query", "")).strip()
            keywords = [str(k).strip() for k in item.get("expected_keywords", []) if str(k).strip()]
            if not query or not keywords:
                continue
            question_set.append({
                "query": query,
                "question_type": str(item.get("question_type", "fact")),
                "expected_keywords": keywords,
            })
        if not question_set:
            raise ValueError("过滤后无有效问题")
        log_info("动态测评", f"动态生成完成，共{len(question_set)}道问题")
        return question_set
    except Exception as e:
        log_warn("动态测评", f"LLM生成失败，降级为规则生成：{e}")
        return _fallback_question_set(topic, effective_ref, count)