# utils/json_repair.py LLM JSON 输出容错解析器
# 功能：修复 LLM 输出 JSON 的常见结构崩坏问题
#   - 全角逗号/冒号（，：）
#   - 字段之间缺少逗号
#   - 字符串值内包含原始换行
#   - 多余/重复引号
# 设计：逐步修复，每步尝试 json.loads，成功即返回；全部失败抛 ValueError
#       调用方应捕获异常并记录原始 LLM 内容（本模块不负责日志，保持纯粹）
import json
import re


def _scan_replace(text: str, replace_map: dict, collapse_ws: bool = False) -> str:
    """
    逐字符扫描文本，仅对【引号外】内容做替换：
    - replace_map: {旧字符: 新字符}
    - collapse_ws: 引号外的换行/制表是否压缩为单个空格
    """
    out = []
    in_str = False
    escaped = False
    for ch in text:
        if in_str:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        # 引号外
        if ch == '"':
            in_str = True
            out.append(ch)
        elif collapse_ws and ch in ("\n", "\r", "\t"):
            # 压缩为单个空格（保持字段之间可读间隔）
            if out and out[-1] != " ":
                out.append(" ")
        else:
            out.append(replace_map.get(ch, ch))
    return "".join(out)


def _escape_newlines_in_strings(text: str) -> str:
    """字符串值内出现的原始换行/回车替换为字面 \\n（避免 JSON 解析失败）"""
    out = []
    in_str = False
    escaped = False
    for ch in text:
        if in_str:
            if escaped:
                out.append("\\" + ch)
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
                out.append(ch)
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            else:
                out.append(ch)
            continue
        if ch == '"':
            in_str = True
        out.append(ch)
    return "".join(out)


def _fix_missing_commas(text: str) -> str:
    """修复字段间缺少逗号：值结束符（} / ] / 字符串闭合 / 数字 / true/false）后紧跟键开始或值开始"""
    # 值结束符后跟 键/值字符串开始
    text = re.sub(r'([}"])\s*(")', r"\1,\2", text)
    # 数字后跟字符串开始
    text = re.sub(r"([0-9])\s*(\")", r"\1,\2", text)
    # true/false/null 后跟字符串开始
    text = re.sub(r"(true|false|null)\s*(\")", r"\1,\2", text)
    return text


def robust_json_loads(text: str) -> dict:
    """
    容错解析 LLM 输出的 JSON 对象（最佳努力，逐步修复）。
    Args:
        text: LLM 原始输出（可能带 Markdown 围栏、前后废话）
    Returns:
        解析后的 dict
    Raises:
        ValueError: 所有修复策略均失败（调用方应记录原文并降级）
    """
    raw = text.strip()
    # 去除 Markdown 代码围栏
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError(f"未找到JSON对象。原文({len(raw)}字)：{raw[:300]}")
    candidate = raw[start:end]

    # 0) 直接解析
    try:
        return json.loads(candidate)
    except Exception:
        pass

    # 1) 引号外全角转半角 + 引号外压缩空白
    fixed = _scan_replace(candidate, {"，": ",", "：": ":"}, collapse_ws=True)
    try:
        return json.loads(fixed)
    except Exception:
        pass

    # 2) 多余引号：连续引号去重（必须先于补逗号，否则 ""第一句"" 会被误补逗号）
    fixed2 = re.sub(r'""+', '"', fixed)
    try:
        return json.loads(fixed2)
    except Exception:
        pass

    # 3) 修复字段间缺失逗号
    fixed3 = _fix_missing_commas(fixed2)
    try:
        return json.loads(fixed3)
    except Exception:
        pass

    # 4) 字符串值内原始换行 → 字面 \\n
    fixed4 = _escape_newlines_in_strings(fixed3)
    try:
        return json.loads(fixed4)
    except Exception:
        pass

    raise ValueError(f"JSON修复失败。处理后内容({len(fixed4)}字)：{fixed4[:300]}")
