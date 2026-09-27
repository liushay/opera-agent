# agent/intent_guard.py 意图校验守卫（问题3修复）
# 功能：
#   - 主管节点任务分派前的意图校验：只允许用户明确表达的任务，禁止自动追加多余功能
#   - 知识问答/介绍类请求只允许 search_worker，绝不派生成类 worker（如闯关/画像）
#   - 历史对话过滤：只保留最近用户消息，杜绝旧AI工具输出混入本次生成上下文
# 设计原则：
#   - 纯函数实现，不依赖任何重型模块（不导入 Ollama/Chroma/rag），可被单测快速验证
#   - 不改变记忆文件结构，仅在业务调用层增加校验逻辑
from typing import Any, Dict, List, Optional

from utils.logger import log_info, log_warn


# 各 Worker 对应的用户请求关键词（用于规则化意图校验）
WORKER_KEYWORDS = {
    "quiz_worker": ["闯关", "答题", "出题", "题目", "测试题", "考题", "クイズ", "quiz"],
    "guide_worker": ["学习", "课程", "路线", "学戏", "教程", "学习计划", "guide"],
    # face_worker 需要"生成/绘制/设计/专属画像"等动作信号；
    # 仅"介绍脸谱/什么是脸谱"属于知识问答，应走 search_worker
    "face_worker": ["生成脸谱", "画脸谱", "绘制脸谱", "脸谱画像", "专属脸谱", "设计脸谱", "脸谱设计", "生成画像", "脸谱生成"],
    "lyrics_worker": ["戏词", "唱词", "台词", "歌词", "解剖", "lyrics"],
    "character_worker": ["对谈", "聊聊", "对话", "和谁", "character", "聊天", "talk"],
    "search_worker": [],  # 检索为兜底，不限制
}


# 知识问答/介绍类前缀：命中的 query 优先走 search_worker，绝不派生成类 worker
# 注意：这些前缀仅作为参考信号，不能粗暴判定——需结合上下文灵活分析
KNOWLEDGE_QUERY_PREFIXES = (
    "介绍", "什么是", "讲讲", "科普", "解释", "说说", "有哪些",
    "介绍一下", "详细说说", "简述", "概述", "是什么",
)

# 角色扮演/对谈信号词：用户明确要求以某角色口吻回答
ROLE_PLAY_SIGNALS = (
    "的口吻", "的角色", "的身份", "的语气", "的口氣",
    "以", "用", "扮演", "假装", "作为",
)

# 复合需求连接词：用户通过这类词组合多个独立需求
COMPOUND_CONNECTORS = ("并", "同时", "另外", "还有", "以及", "顺便", "而且", "并且", "与此同时")


def _has_role_play_signal(query: str) -> bool:
    """
    检测用户请求是否包含角色扮演/对谈需求。
    覆盖多种常见表达模式，不局限于固定模板。

    模式1：以/用XX的口吻/角色/身份/语气
    模式2：扮演/假装/假设/充当/变成 + 角色
    模式3：你现在是/现在你是/你是（角色代入）
    模式4：作为XX（回答/回复/说话/来说）
    模式5：你当/你来当/你扮演/你假装 + 角色
    """
    q = query.strip()

    # 模式1：以/用XX的口吻/角色/身份/语气
    # 同时支持"角色口吻"（无"的"连接）和"XX角色口吻"等变体
    for prefix in ("以", "用"):
        if prefix in q:
            for signal in ("的口吻", "的角色", "的身份", "的语气", "的口氣"):
                if signal in q:
                    return True
            # 变体：以/用XX + 角色口吻/角色身份（无"的"连接）
            # 例如："用昆曲中的杜丽娘角色口吻回答"
            if "角色" in q:
                for tone in ("口吻", "身份", "语气", "口氣"):
                    if tone in q:
                        return True

    # 模式2：角色代入动词 + 角色名
    role_adoption_patterns = (
        "扮演", "假装你是", "假装是", "假设你是", "假设是",
        "你现在是", "现在你是", "你变成", "变成",
        "当作你是", "当做你是", "充当", "你来扮演",
        "你扮演", "你假装", "你当", "你来当",
    )
    for pattern in role_adoption_patterns:
        if pattern in q:
            return True

    # 模式3：作为XX（回答/回复/说话/来说/来讲）
    if "作为" in q:
        for action in ("回答", "回复", "说话", "来说", "来讲", "说", "讲"):
            if action in q:
                return True

    # 模式4：你是XX（角色代入，排除"你是谁"、"你是什么"等疑问句）
    if "你是" in q and "你是谁" not in q and "你是什么" not in q:
        # 进一步确认：不是在问"你是..."的疑问句
        if not q.endswith("吗") and not q.endswith("么") and "什么" not in q[q.index("你是"):q.index("你是")+10]:
            return True

    return False


def _has_compound_connector(query: str) -> bool:
    """检测用户请求是否包含复合需求连接词"""
    return any(connector in query for connector in COMPOUND_CONNECTORS)


def is_compound_query(query: str) -> bool:
    """
    判断用户请求是否为复合需求（包含多个独立功能需求）。
    判断标准：
    1. 包含复合连接词（并/同时/另外/还有/以及 等）
    2. 同时包含知识问答信号 + 角色扮演信号
    3. 同时包含知识问答信号 + 其他功能信号（戏词/脸谱/闯关等）
    """
    # 规则1：有复合连接词
    if _has_compound_connector(query):
        return True
    # 规则2：同时有知识问答信号 + 角色扮演信号
    has_knowledge = _has_knowledge_signal(query) or is_knowledge_query(query)
    has_role = _has_role_play_signal(query)
    if has_knowledge and has_role:
        return True
    return False


def is_knowledge_query(query: str) -> bool:
    """
    判断用户请求是否为纯知识问答/介绍类（灵活判断，不粗暴）。
    重要：如果用户同时包含角色扮演需求，则不判定为纯知识问答。
    """
    q = query.strip()
    # 如果用户有角色扮演需求，不是纯知识问答
    if _has_role_play_signal(q):
        return False
    # 如果用户有复合需求连接词，不是纯知识问答
    if _has_compound_connector(q):
        return False
    return any(q.startswith(prefix) or f" {prefix}" in q for prefix in KNOWLEDGE_QUERY_PREFIXES)


# 图像生成诉求关键词（用于规则化识别"用户需要图片"的诉求）：
# 修复(Bug)：用户"生成一张越剧有关的脸谱"因"生成脸谱"不连续而漏匹配 face_worker。
# 这里放宽为动作动词 + 图像名词 的组合识别，既覆盖连续短语，也覆盖分隔场景。
IMAGE_INTENT_VERBS = ("生成", "画", "绘制", "制作", "设计", "创作", "做一个", "来一个", "弄一个", "给我")
IMAGE_INTENT_NOUNS = ("图片", "图画", "画像", "脸谱", "图像", "图", "照片", "海报", "画作", "图样", "插画")
# 图像名词单独出现即可触发（如"脸谱"本身即指图像）
IMAGE_INTENT_NOUNS_ALONE = ("脸谱", "画像", "图片", "图画", "图像", "照片", "海报", "图样", "插画")

# 知识问答信号词：当 query 包含这些词时，大概率是知识问答而非图像生成诉求
# 例如"脸谱有哪些颜色"、"画像的起源" — 用户是在问知识，不是要生成图片
_KNOWLEDGE_SIGNAL_WORDS = (
    "有哪些", "是什么", "怎么样", "为什么", "如何", "怎么",
    "什么样", "颜色", "起源", "历史", "代表", "含义", "种类",
    "分类", "特征", "特点", "典故", "传说", "故事", "作用",
    "象征", "区别", "对比", "比较", "由来", "发展", "流派",
    "介绍", "讲讲", "科普", "解释", "说说", "简述", "概述",
    "意思", "定义", "概念", "背景", "文化", "艺术", "传统",
)


def _has_knowledge_signal(query: str) -> bool:
    """检查 query 是否包含知识问答信号词（如疑问词、解释类词汇）"""
    q = query.strip().lower()
    return any(word in q for word in _KNOWLEDGE_SIGNAL_WORDS)


def is_image_intent(query: str) -> bool:
    """
    判断用户请求是否包含"生成图片/画图/画像"类诉求。
    规则：
      1. 包含动作动词 + 图像名词的组合（允许中间有其他字，如"生成一张越剧有关的脸谱"）
      2. 单独包含强图像名词（脸谱/画像/图片等）且不属于知识问答
    该函数供 task 拆解 / 输出校验 / 前端提示 共用，保持纯函数可单测。
    """
    q = query.strip().lower()
    if not q:
        return False
    # 纯知识问答（介绍/什么是脸谱/有哪些/怎么等）不算图像生成诉求
    if is_knowledge_query(query) or _has_knowledge_signal(query):
        return False
    for verb in IMAGE_INTENT_VERBS:
        if verb not in q:
            continue
        for noun in IMAGE_INTENT_NOUNS:
            if noun in q:
                return True
    # 强图像名词直接触发：但需排除知识问答信号
    # （"脸谱有哪些颜色"、"画像的起源"等是知识问答，不是图像生成）
    for noun in IMAGE_INTENT_NOUNS_ALONE:
        if noun in q:
            # 二次确认：不含知识问答信号词才触发
            log_info("意图校验", f"检测到图像名词[{noun}]，确认无知识问答信号后触发图像生成意图")
            return True
    return False


def validate_worker_against_query(worker: str, query: str) -> bool:
    """
    意图校验：用户本轮真实需求中是否包含该 worker 对应的关键词。
    若用户没有明确表达该任务需求，则禁止派遣该 worker（防止自动追加闯关等多余任务）。

    重要改进：不再死板依赖关键词列表，而是结合语义信号灵活判断。
    - character_worker：只要有角色扮演信号（"以XX的口吻/语气/角色/身份"）即放行
    - face_worker：只要有图像生成诉求即放行
    - 其他 worker：关键词匹配 + 知识问答降级
    """
    keywords = WORKER_KEYWORDS.get(worker, [])
    if not keywords:
        return True  # 无关键词约束的 worker（如 search_worker）放行
    query_lower = query.lower()

    # 知识问答/介绍类请求：只允许 search_worker，禁止生成类 worker
    if is_knowledge_query(query) and worker != "search_worker":
        log_info("意图校验", f"用户请求为知识问答/介绍类，禁止派遣[{worker}]")
        return False

    # 灵活放行：character_worker 只要有角色扮演信号即放行
    # 例："以穆桂英的语气回答你好"、"用杜丽娘的口吻回复" — 不含关键词但明显是角色扮演
    if worker == "character_worker" and _has_role_play_signal(query):
        log_info("意图校验", f"用户请求包含角色扮演信号，放行[character_worker]")
        return True

    # 灵活放行：face_worker 只要有图像生成诉求即放行
    # 即便连续关键词未命中（如"生成脸谱"被"一张越剧有关"分隔）
    if worker == "face_worker" and is_image_intent(query):
        log_info("意图校验", f"用户请求包含图像生成诉求，放行[face_worker]")
        return True

    # 关键词兜底：检查是否命中预定义的关键词列表
    return any(kw.lower() in query_lower for kw in keywords)


def filter_tasks_by_intent(task_data: list, query: str, intent: Optional[Dict[str, Any]] = None) -> list:
    """
    意图校验核心函数：
    1. 每个任务必须通过关键词校验（用户必须明确表达该需求）
    2. 复合需求(is_compound=true)：保留多个对应的 worker，按顺序执行
    3. 单一需求：若模型输出了多个 worker，只保留与用户核心任务最匹配的一个
    4. 用户未提出闯关/出题时，绝不保留 quiz_worker
    5. 知识问答/介绍类请求被误判为生成类worker时，降级为 search_worker（知识检索兜底）
    6. 修复(Bug)：用户有图像生成诉求（生成图片/画图/生成脸谱）时，
       无论模型是否误判为 search_worker，都必须强制派发 face_worker（图像工具调用），
       禁止只派文本生成任务。
    7. 复合需求支持：当 intent.is_compound=true 时，根据 sub_tasks 构建多 worker 任务列表
    """
    # ===== 修复(Bug)：图像诉求最高优先级，强制派发 face_worker =====
    if is_image_intent(query):
        log_info("意图校验", f"用户请求包含图像生成诉求，强制派发[face_worker]：{query}")
        return [{"worker": "face_worker", "task": query, "params": {"preferences": query}}]

    # ===== 复合需求处理：支持多 worker 分派 =====
    # 优先级1：LLM 意图解析成功且标注了 is_compound
    if intent and intent.get("is_compound"):
        sub_tasks = intent.get("sub_tasks", [])
        if sub_tasks:
            # worker 名称映射：LLM 输出的短名 → 实际 worker 名
            _WORKER_NAME_MAP = {
                "search": "search_worker",
                "search_worker": "search_worker",
                "character": "character_worker",
                "character_worker": "character_worker",
                "chat": "character_worker",  # chat 角色对谈 → character_worker
                "lyrics": "lyrics_worker",
                "lyrics_worker": "lyrics_worker",
                "quiz": "quiz_worker",
                "quiz_worker": "quiz_worker",
                "guide": "guide_worker",
                "guide_worker": "guide_worker",
                "face": "face_worker",
                "face_worker": "face_worker",
            }
            result = []
            for st in sub_tasks:
                raw_worker = st.get("worker", "search_worker")
                worker = _WORKER_NAME_MAP.get(raw_worker, raw_worker)
                task = st.get("task", query)
                result.append({
                    "worker": worker,
                    "task": task,
                    "params": {"sub_task": task}
                })
            log_info("意图校验", f"复合需求（LLM解析）：分派 {len(result)} 个 worker：{[r['worker'] for r in result]}")
            return result

    # 优先级2：规则兜底 — 当 intent 解析失败但规则检测到复合需求时，自动构造多 worker 分派
    # 场景：intent JSON 解析失败 → is_compound 丢失，但规则层能检测到"知识问答+角色扮演"等复合信号
    if (not intent or not intent.get("is_compound")) and is_compound_query(query):
        log_info("意图校验", f"规则检测到复合需求（intent解析可能失败），构造多worker分派：{query}")
        compounds = []
        # 知识问答/介绍需求 → search_worker
        has_knowledge = _has_knowledge_signal(query) or any(
            query.strip().startswith(p) or f" {p}" in query
            for p in KNOWLEDGE_QUERY_PREFIXES
        )
        if has_knowledge:
            compounds.append({"worker": "search_worker", "task": query, "params": {}})
        # 角色扮演/对谈需求 → character_worker
        if _has_role_play_signal(query):
            compounds.append({"worker": "character_worker", "task": query, "params": {"sub_task": "角色扮演对话"}})
        # 戏词解剖需求 → lyrics_worker
        if any(kw in query for kw in WORKER_KEYWORDS.get("lyrics_worker", [])):
            compounds.append({"worker": "lyrics_worker", "task": query, "params": {"sub_task": "戏词解剖"}})
        if len(compounds) >= 2:
            log_info("意图校验", f"复合需求（规则兜底）：分派 {len(compounds)} 个 worker：{[c['worker'] for c in compounds]}")
            return compounds
        # 只有一个需求时，直接返回该任务
        if compounds:
            return compounds

    if not task_data:
        # 无任务但用户是知识问答/介绍类 → 降级为检索（保证能回答）
        if is_knowledge_query(query):
            log_info("意图校验", "模型未分派任务但用户为知识问答，降级为检索工人")
            return [{"worker": "search_worker", "task": query, "params": {}}]
        return []

    # 第一步：过滤掉用户未明确表达的任务
    permitted = []
    for t in task_data:
        worker = t.get("worker", "")
        if worker == "none":
            continue
        if worker not in WORKER_KEYWORDS:
            continue
        if not validate_worker_against_query(worker, query):
            log_info("意图校验", f"用户未表达[{worker}]需求，移除该任务（防止追加多余功能）")
            continue
        permitted.append(t)

    # 第二步：复合需求时保留多个 worker，否则只保留第一个
    if is_compound_query(query):
        log_info("意图校验", f"复合需求，保留 {len(permitted)} 个 worker")
        # 保持 permitted 原样，不缩减
    elif len(permitted) > 1:
        log_info("意图校验", f"模型输出多个任务，仅保留最匹配一个：{permitted[0]}")
        permitted = permitted[:1]

    # 第三步：全部被过滤掉但用户是知识问答/介绍类 → 降级为检索（保证能回答）
    if not permitted and is_knowledge_query(query):
        log_info("意图校验", "生成类任务被过滤且用户为知识问答，降级为检索工人")
        return [{"worker": "search_worker", "task": query, "params": {}}]

    return permitted


def filter_history_recent(history: list, max_user_msgs: int = 2) -> str:
    """
    历史对话过滤：只提取最近用户消息摘要，过滤掉旧AI工具输出。
    绝不让上一次的闯关题目/文献/检索结果混入本次生成上下文。
    """
    if not history:
        return ""
    # 尝试从 BaseMessage / dict 中提取用户消息
    user_msgs = []
    for m in history:
        # langchain BaseMessage
        if hasattr(m, "type") and m.type == "human":
            user_msgs.append(getattr(m, "content", ""))
            continue
        if isinstance(m, dict):
            role = m.get("type", "") or m.get("role", "")
            content = m.get("content", "")
            if role in ("human", "user"):
                user_msgs.append(content)
    if not user_msgs:
        return ""
    recent = user_msgs[-max_user_msgs:]
    return "\n".join(f"- 用户之前问过：{str(content)[:100]}" for content in recent)


def normalize_face_image_result(result: dict) -> dict:
    """
    多Agent脸谱结果图片URL规范化（问题2修复）：
    将 generate_face_profile 返回的本地磁盘路径 image_url 规范化为前端可访问的 /static URL，
    并补齐 local_file_path / real_file_path 字段（与独立脸谱页面 opera_routes.py 后处理一致）。
    保证多Agent输出与单页面脸谱输出格式一致、图片可在前端渲染。
    """
    if not result or "error" in result:
        return result
    image_url = result.get("image_url", "")
    local_file_path = ""
    if image_url and not image_url.startswith(("http://", "https://")):
        norm = image_url.replace("\\", "/")
        local_file_path = norm
        if "/literature_output/" in norm:
            rel = norm.split("/literature_output/", 1)[1]
        elif norm.startswith("./literature_output/"):
            rel = norm[len("./literature_output/"):]
        else:
            import os as _os
            rel = _os.path.basename(norm)
        result["image_url"] = f"/static/{rel}"
        result["local_file_path"] = local_file_path
        result["real_file_path"] = local_file_path
        log_info("多Agent脸谱工人", f"图片URL规范化：{image_url} -> {result['image_url']}")
    return result


def build_face_reply_text(face_result: dict, query: str) -> str:
    """
    构造 face_worker 的结构化回复文本（问题2修复）：
    - 使用 Markdown 图片标记（![脸谱](/static/xxx)）使前端 st.markdown 能直接渲染图片
    - 完整保留 image_url / local_file_path / real_file_path / image_status 等图片相关字段
    - 与独立脸谱页面字段一致，绝不用示例占位文本
    """
    face_name = face_result.get("face_name", "无名脸谱")
    color = face_result.get("color", "红")
    color_meaning = face_result.get("color_meaning", "")
    pattern = face_result.get("pattern", "")
    pattern_meaning = face_result.get("pattern_meaning", "")
    matching_char = face_result.get("matching_character", "")
    personality = face_result.get("personality_text", "")
    share_card = face_result.get("share_card", "")
    image_status = face_result.get("image_status", "text_only")
    image_url = face_result.get("image_url", "")

    lines = []
    lines.append(f"## 🎭 你的专属脸谱：{face_name}\n")
    lines.append(f"- **主色**：{color}（{color_meaning}）")
    if pattern:
        lines.append(f"- **图案设计**：{pattern}（{pattern_meaning}）")
    if matching_char:
        lines.append(f"- **气质人物**：{matching_char}")
    if personality:
        lines.append(f"\n### 🧬 人格解读\n{personality}")
    if share_card:
        lines.append(f"\n### 📤 分享卡片\n{share_card}")

    # 图片渲染（image_status=generated 且 image_url 非空时）
    if image_status == "generated" and image_url:
        lines.append("\n### 🖼️ 脸谱图像（即梦AI生成）")
        lines.append(f"![{face_name}]({image_url})")
        lines.append(f"\n- 图片访问地址：`{image_url}`")
        lines.append(f"- 本地文件路径：`{face_result.get('local_file_path', face_result.get('real_file_path', ''))}`")
        lines.append(f"- image_status：`{image_status}`")
    else:
        lines.append("\n> 当前为文本版脸谱（image_status=text_only），未生成图像。")

    result_text = "\n".join(lines)
    log_info("多Agent汇总节点", f"face_worker 结构化直出（含图片URL），长度{len(result_text)}字")
    return result_text
