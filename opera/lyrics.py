# opera/lyrics.py 戏词解剖室
# 功能：对用户输入的经典戏词进行多维度解读
#   - 逐句白话翻译
#   - 典故/出处考据（检索知识库）
#   - 人物心境解读
#   - 唱腔段式标注（行当/曲牌/声腔）
#   - 生成"戏词品鉴卡"文案
import json
from typing import Dict, Any

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

import config
from rag.vectorstore import hybrid_retrieve
from utils.logger import log_info, log_warn, log_error
from utils.json_repair import robust_json_loads


def _get_llm():
    """复用主LLM（qwen2.5:3b），与项目其他Agent一致"""
    return ChatOllama(model=config.LLM_MODEL, temperature=0.2)


def _search_lyrics_kb(lyric_text: str) -> str:
    """检索知识库中与戏词相关的资料（出处/剧情/人物背景）"""
    try:
        docs = hybrid_retrieve(lyric_text)
        if not docs:
            return ""
        return "\n".join(
            [f"资料片段：{doc.page_content}（来源：{doc.metadata}）" for doc in docs]
        )
    except Exception as e:
        log_warn("戏词解剖室", f"知识库检索失败，降级为无资料：{e}")
        return ""


def annotate_lyrics(lyrics: str) -> Dict[str, Any]:
    """
    戏词解剖：对一段戏词生成结构化解读
    Args:
        lyrics: 用户输入的戏词原文
    Returns:
        {
            "original": 原文,
            "annotation": 逐句白话翻译,
            "allusion": 典故出处,
            "character_mood": 人物心境解读,
            "singing_style": 唱腔段式标注,
            "appreciation": 品鉴文案,
            "sources": 检索到的参考资料
        }
    """
    log_info("戏词解剖室", f"开始解剖戏词：{lyrics[:50]}...")
    lyric_text = lyrics.strip()
    if not lyric_text:
        return {"error": "戏词不能为空"}

    # 1. 检索知识库参考资料
    kb_material = _search_lyrics_kb(lyric_text)
    kb_part = ""
    if kb_material:
        kb_part = f"\n知识库参考资料：\n{kb_material}"

    llm = _get_llm()
    prompt = f"""
你是资深戏曲品鉴家。请对以下戏词进行全方位解剖，输出严格JSON。

戏词原文：
{lyric_text}
{kb_part}

请按以下结构输出JSON（字段必须齐全）：
{{
  "annotation": "逐句白话翻译，每句一行：原文 → 白话",
  "allusion": "典故与出处考据，含剧目/人物/历史背景",
  "character_mood": "人物此刻的心境与情感解读",
  "singing_style": "唱腔段式标注（行当/声腔/曲牌/板式，若可推断则标注，否则写'需结合剧目确认'）",
  "appreciation": "一段60-100字的品鉴文案，语言优美有韵味，可作分享卡片文案"
}}
【JSON硬性规范】
1. 所有字段名和字符串值必须使用英文双引号包裹
2. 字段之间必须使用英文半角逗号分隔；禁止使用全角逗号（，）或全角冒号（：）
3. 字符串值内禁止包含原始换行符；如需换行请使用转义符 \\n
4. 禁止在值首尾添加多余引号
5. 只输出一个JSON对象，不要输出```围栏，不要有任何多余文字
"""
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        # ===== 修复：使用容错JSON解析器（修复全角标点/缺逗号/值内换行/多余引号） =====
        try:
            data = robust_json_loads(content)
        except Exception as parse_err:
            # 记录原始 LLM 返回内容（便于排查），再降级
            log_error(
                "戏词解剖室",
                f"JSON解析失败，原始LLM返回内容({len(content)}字)：{content[:600]}",
                parse_err,
            )
            data = {
                "annotation": "",
                "allusion": "",
                "character_mood": "",
                "singing_style": "",
                "appreciation": f"（戏词解读生成失败，原文如下）\n{lyric_text}",
            }
    except Exception as e:
        log_error("戏词解剖室", f"LLM调用失败：{e}", e)
        return {"error": f"解读生成失败：{str(e)}", "original": lyric_text}

    result = {
        "original": lyric_text,
        "annotation": data.get("annotation", ""),
        "allusion": data.get("allusion", ""),
        "character_mood": data.get("character_mood", ""),
        "singing_style": data.get("singing_style", ""),
        "appreciation": data.get("appreciation", ""),
        "sources": kb_material or "知识库未检索到相关参考资料",
    }
    log_info("戏词解剖室", "戏词解剖完成")
    return result