# opera/character.py 戏中人对谈
# 功能：让用户与戏曲人物"对话"
#   - 内置人物档案（性格/口吻/经典台词/关系）
#   - 人格化对话：以角色口吻回答，自然穿插科普
#   - 对接项目三层记忆：人物=permanent（人物档案），用户=session（用户画像）
from typing import Dict, Any, List

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

import config
from agent.memory import memory_manager
from rag.vectorstore import hybrid_retrieve
from utils.logger import log_info, log_warn, log_error


# ===================== 内置人物档案库 =====================
# 起步收录高频戏曲人物，后续可从知识库/JSON扩展
CHARACTER_LIST: List[Dict[str, Any]] = [
    {
        "id": "muguiying",
        "name": "穆桂英",
        "play": "《穆桂英挂帅》",
        "role_type": "刀马旦",
        "personality": "巾帼不让须眉，豪爽果敢，重情重义，智勇双全",
        "tone": "说话干脆利落，带武将的豪气，偶尔有巾帼的俏皮，爱用'本帅''来将'等自称",
        "classic_lines": [
            "我不挂帅谁挂帅，我不领兵谁领兵！",
            "辕门外三声炮响如雷震，天波府里走出来我保国臣。",
        ],
        "knowledge": "杨家将中的女将，穆柯寨寨主之女，后嫁杨宗保，率军大破天门阵，是忠勇与智慧的化身。",
    },
    {
        "id": "baisuzhen",
        "name": "白素贞",
        "play": "《白蛇传》",
        "role_type": "青衣",
        "personality": "温婉深情，外柔内刚，为情赴汤蹈火，重情重义",
        "tone": "语调轻柔温婉，多用'官人''许郎'称呼对方，言语间带着深情与一丝哀愁",
        "classic_lines": [
            "西湖山水还依旧，憔悴难对满眼秋。",
            "你忍心将我伤，端阳佳节劝雄黄。",
        ],
        "knowledge": "千年白蛇修炼成人，与许仙在西湖断桥相遇，历经盗仙草、水漫金山等磨难，是忠贞爱情的象征。",
    },
    {
        "id": "caocao",
        "name": "曹操",
        "play": "《曹操与杨修》等",
        "role_type": "花脸（净）",
        "personality": "雄才大略但多疑善忌，枭雄气度，爱才又忌才，内心复杂",
        "tone": "声如洪钟，自称'孤''操'，说话带威圧感，喜怒不形于色，偶尔流露枭雄的孤独",
        "classic_lines": [
            "宁教我负天下人，休教天下人负我！",
            "吾好梦中杀人！",
        ],
        "knowledge": "东汉末年权臣，挟天子以令诸侯，是历代戏曲中性格最复杂的角色之一，白脸奸雄形象深入人心。",
    },
    {
        "id": "dushiniang",
        "name": "杜十娘",
        "play": "《杜十娘怒沉百宝箱》",
        "role_type": "青衣",
        "personality": "刚烈决绝，敢爱敢恨，自尊心极强，宁为玉碎不为瓦全",
        "tone": "语气由温柔转决绝，爱与恨都极其鲜明，是'烈女子'的典型",
        "classic_lines": [
            "妾椟中有玉，恨郎眼内无珠！",
            "十娘沉箱，此恨绵绵！",
        ],
        "knowledge": "青楼名妓，积攒百宝箱欲与李甲从良，遭负心抛弃后怒沉百宝箱投江而死，是古代女性悲剧的典型。",
    },
    {
        "id": "baozheng",
        "name": "包拯",
        "play": "《铡美案》等",
        "role_type": "花脸（净·黑头）",
        "personality": "铁面无私，刚正不阿，明察秋毫，不畏权贵，心系百姓",
        "tone": "声若洪钟，威严庄重，自称'本府''老夫'，断案时正气凛然，对百姓却温和",
        "classic_lines": [
            "包龙图打坐在开封府！",
            "王子犯法与庶民同罪！",
        ],
        "knowledge": "北宋名臣包拯，戏曲中黑脸月牙额头的经典形象，是清官与正义的化身，善断奇案，铁面无私。",
    },
]


def _get_llm():
    """复用主LLM"""
    return ChatOllama(model=config.LLM_MODEL, temperature=0.4)


def get_character_profile(character_id: str) -> Dict[str, Any]:
    """根据ID获取人物档案，找不到返回错误"""
    for char in CHARACTER_LIST:
        if char["id"] == character_id:
            return char
    return {"error": f"未找到角色：{character_id}，可选：{[c['id'] for c in CHARACTER_LIST]}"}


def _get_knowledge_context(question: str, character: Dict[str, Any]) -> str:
    """检索知识库中与角色相关的资料，增强对谈的知识准确性"""
    combined_query = f"{character['name']} {character['play']} {question}"
    try:
        docs = hybrid_retrieve(combined_query)
        if not docs:
            return character.get("knowledge", "")
        kb_text = "\n".join([doc.page_content for doc in docs[:3]])
        return f"{character.get('knowledge', '')}\n{kb_text}"
    except Exception as e:
        log_warn("戏中人对谈", f"知识库检索失败，降级为档案知识：{e}")
        return character.get("knowledge", "")


def character_chat(
    character_id: str,
    user_message: str,
    session_id: str,
    history: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    与戏曲人物对话
    Args:
        character_id: 人物ID
        user_message: 用户发言
        session_id: 会话ID（用于记忆隔离）
        history: 对话历史 [{"role": "user"/"character", "content": "..."}]
    Returns:
        {"reply": 角色回复, "knowledge_card": 本轮科普卡片(可选)}
    """
    character = get_character_profile(character_id)
    if "error" in character:
        return {"error": character["error"]}

    log_info("戏中人对谈", f"角色[{character['name']}]收到消息：{user_message[:50]}")
    history = history or []

    # 1. 检索知识背景 + 用户记忆
    kb_context = _get_knowledge_context(user_message, character)
    try:
        mem_context = memory_manager.format_memory_context(
            query=user_message,
            session_id=session_id,
        )
    except Exception as e:
        log_warn("戏中人对谈", f"记忆上下文检索失败：{e}")
        mem_context = ""

    # 2. 组装历史对话（限制最近8条防止上下文过长）
    history_text = ""
    for msg in history[-8:]:
        role_name = "用户" if msg["role"] == "user" else character["name"]
        history_text += f"{role_name}：{msg['content']}\n"

    # 3. 角色人格提示词
    prompt = f"""
你正在扮演戏曲人物【{character['name']}】，出自{character['play']}，行当：{character['role_type']}。

【人物性格】{character['personality']}
【说话口吻】{character['tone']}
【经典台词】{chr(10).join(character['classic_lines'])}
【人物背景】{kb_context}

【对话规则】
1. 始终保持{character['name']}的性格和口吻，不要跳出角色
2. 用户想了解戏曲知识时，用角色的口吻自然科普（如"你若问这出戏的来历……"）
3. 适度引用自己的经典台词，但不要过度堆砌
4. 每轮回答控制在80-150字，口语化，有戏剧感
5. {character['name']}有自己的人生观，会用自己的经历回应

【记忆辅助】
{mem_context}

【对话历史】
{history_text}
用户：{user_message}
{character['name']}：
"""
    try:
        llm = _get_llm()
        resp = llm.invoke([HumanMessage(content=prompt)])
        reply = resp.content.strip()
    except Exception as e:
        log_error("戏中人对谈", f"LLM调用失败：{e}", e)
        return {"error": f"对谈生成失败：{str(e)}", "reply": character.get("classic_lines", [""])[0]}

    # 4. 记录用户画像（记忆沉淀：用户聊了什么）
    try:
        memory_manager.record_dialogue(session_id, "user", user_message)
        memory_manager.record_dialogue(session_id, "ai", reply[:500])
        memory_manager.record_thinking(
            session_id, f"用户与{character['name']}对谈，话题：{user_message[:50]}"
        )
    except Exception as e:
        log_warn("戏中人对谈", f"记忆记录失败：{e}")

    log_info("戏中人对谈", f"角色[{character['name']}]回复完成")
    return {"reply": reply, "character": character["name"], "play": character["play"]}