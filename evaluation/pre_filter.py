# evaluation/pre_filter.py 快速预筛选器
# 功能：
#   在深度评估（LLM）之前，用纯规则+正则快速判断生成内容是否合格。
#   90%的正常请求直接通过，仅10%存疑的进入深度评估。
#   这不替代深度评估，而是作为第一道低成本过滤器。
# 设计原则：
#   - 纯规则，不调用LLM，耗时 < 50ms
#   - 宁可漏过（false negative）不可误杀（false positive）：只拦截明显有问题的
#   - 每个检查项独立，可单独开关
#   - 与现有的 intent_guard / output_validator / eval_agent 完全解耦，不修改任何现有代码
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from utils.logger import log_info


# ===================== 数据结构 =====================

@dataclass
class PreFilterResult:
    """预筛选结果"""
    passed: bool = True                       # 是否通过预筛选（True=放行，不需要深度评估）
    suspicious: bool = False                  # 是否存疑（True=需要深度评估）
    reasons: List[str] = field(default_factory=list)  # 存疑原因
    severity: str = "none"                    # 严重程度：none / low / medium / high
    metrics: Dict[str, Any] = field(default_factory=dict)  # 详细指标


# ===================== 检查项配置 =====================

# 空回复检测：最小有效长度（字符）
MIN_VALID_LENGTH = 10

# 拒绝类关键词（模型拒绝回答的典型话术）
REJECTION_PATTERNS = [
    r"作为AI.*无法",
    r"对不起.*我无法",
    r"抱歉.*不能",
    r"作为一个AI",
    r"我无法.*生成",
    r"我没有.*能力",
    r"超出.*能力范围",
    r"暂时无法",
    r"我无法回答",
    r"我不确定",
    r"请提供更多",
    r"我无法理解",
]

# 无意义回复模式
MEANINGLESS_PATTERNS = [
    r"^好的[，,。.]?$",
    r"^明白了[，,。.]?$",
    r"^收到[，,。.]?$",
    r"^没问题[，,。.]?$",
    r"^OK[，,。.]?$",
    r"^嗯[，,。.]?$",
]

# 格式完整性检测：未闭合的Markdown代码块/标签
UNCLOSED_FORMAT_PATTERNS = [
    (r"```(?![\s\S]*```)", "未闭合的代码块"),
    (r"\[([^\]]+)\]\((?!\S+\))", "未闭合的Markdown链接"),
    (r"<([a-zA-Z]+)>[\s\S]*?(?!</\1>)", "未闭合的HTML标签"),
]


# ===================== 预筛选器主类 =====================

class FastPreFilter:
    """
    快速预筛选器：纯规则判断，不调用LLM。
    在 eval_agent 深度评估之前运行，拦截明显不合格的生成内容。
    """

    def __init__(self):
        self._rejection_re = [re.compile(p, re.IGNORECASE) for p in REJECTION_PATTERNS]
        self._meaningless_re = [re.compile(p, re.IGNORECASE) for p in MEANINGLESS_PATTERNS]

    def check(
        self,
        user_query: str,
        generated_text: str,
        image_resources: List[Dict[str, Any]] = None,
    ) -> PreFilterResult:
        """
        执行所有预筛选检查。

        Args:
            user_query: 用户原始请求
            generated_text: 生成文本内容
            image_resources: 图片资源列表（可选）

        Returns:
            PreFilterResult: 包含是否通过、存疑原因、严重程度
        """
        result = PreFilterResult()
        text = str(generated_text).strip() if generated_text else ""

        # 检查1：空/过短回复
        if not text or len(text) < MIN_VALID_LENGTH:
            result.passed = False
            result.suspicious = True
            result.reasons.append(f"回复过短（{len(text)}字），可能为空或截断")
            result.severity = "high"
            result.metrics["empty_or_short"] = True
            log_info("预筛选器", f"拦截：回复过短（{len(text)}字）")
            return result

        result.metrics["empty_or_short"] = False

        # 检查2：拒绝类回复
        for i, pat in enumerate(self._rejection_re):
            if pat.search(text):
                result.passed = False
                result.suspicious = True
                result.reasons.append(f"检测到拒绝类话术：{REJECTION_PATTERNS[i]}")
                result.severity = "high"
                result.metrics["rejection_detected"] = True
                log_info("预筛选器", f"拦截：拒绝类话术匹配 {REJECTION_PATTERNS[i]}")
                return result

        result.metrics["rejection_detected"] = False

        # 检查3：无意义回复
        for pat in self._meaningless_re:
            if pat.match(text):
                result.passed = False
                result.suspicious = True
                result.reasons.append("回复内容无意义（仅寒暄/确认词）")
                result.severity = "medium"
                result.metrics["meaningless"] = True
                log_info("预筛选器", "拦截：无意义回复")
                return result

        result.metrics["meaningless"] = False

        # 检查4：格式完整性
        for pattern, desc in UNCLOSED_FORMAT_PATTERNS:
            if re.search(pattern, text):
                result.suspicious = True
                result.reasons.append(f"格式不完整：{desc}")
                result.severity = "low"
                result.metrics["format_incomplete"] = True
                log_info("预筛选器", f"存疑：格式不完整 - {desc}")
                break
        else:
            result.metrics["format_incomplete"] = False

        # 检查5：关键词覆盖（用户query中的关键实体是否在回复中出现）
        query_entities = self._extract_entities(user_query)
        if query_entities:
            coverage = self._check_entity_coverage(text, query_entities)
            result.metrics["entity_coverage"] = coverage
            if coverage < 0.3 and len(query_entities) >= 3:
                # 用户提到3个以上实体，但回复覆盖不到30%，存疑
                result.suspicious = True
                result.reasons.append(
                    f"关键实体覆盖率低（{coverage:.0%}），用户提到的关键概念可能未被响应"
                )
                result.severity = "medium"
                log_info("预筛选器", f"存疑：实体覆盖率={coverage:.0%}，entities={query_entities}")
        else:
            result.metrics["entity_coverage"] = 1.0

        # 检查6：图像诉求（用户要图但没图 → 高严重度拦截）
        from agent.intent_guard import is_image_intent
        if is_image_intent(user_query):
            image_resources = image_resources or []
            has_real_image = any(
                r.get("type") == "image"
                and r.get("image_status") == "generated"
                and r.get("image_url")
                for r in image_resources
            )
            if not has_real_image:
                result.passed = False
                result.suspicious = True
                result.reasons.append("用户需要图片，但生成结果没有真实图片资源")
                result.severity = "high"
                result.metrics["image_missing"] = True
                log_info("预筛选器", "拦截：图像缺失")
                return result
        result.metrics["image_missing"] = False

        # 综合判断
        if result.suspicious:
            result.passed = False
        else:
            result.passed = True
            log_info("预筛选器", "快速预筛选通过，跳过深度评估")

        return result

    # ===================== 辅助方法 =====================

    def _extract_entities(self, query: str) -> List[str]:
        """
        从用户query中提取关键实体（纯规则，不调LLM）。
        提取：书名号内容、引号内容、中文专有名词（2-4字的不常见词）。
        """
        entities = []
        # 书名号内容
        book_entities = re.findall(r'《([^》]+)》', query)
        entities.extend(book_entities)
        # 引号内容
        quote_entities = re.findall(r'[""]([^""]+)[""]', query)
        entities.extend(quote_entities)
        # 去除常见停用词后的2-4字词（作为实体候选）
        # 简单策略：提取被问到的关键词
        stopwords = {"请问", "我想", "可以", "什么", "怎么", "如何", "为什么", "有哪些",
                     "介绍一下", "讲讲", "说说", "帮我", "给我", "能否", "能不能",
                     "一个", "一下", "这个", "那个", "是", "的", "了", "吗", "呢", "吧"}
        # 用jieba分词提取可能的实体（如果jieba可用）
        try:
            import jieba
            words = jieba.lcut(query)
            for w in words:
                w = w.strip()
                if len(w) >= 2 and w not in stopwords and not w.isdigit() and not all(c in '，。！？、；：""''（）\n\r\t ' for c in w):
                    if w not in entities:
                        entities.append(w)
        except ImportError:
            pass

        return entities[:10]  # 最多10个实体

    def _check_entity_coverage(self, text: str, entities: List[str]) -> float:
        """检查关键实体在回复中的覆盖率"""
        if not entities:
            return 1.0
        covered = sum(1 for e in entities if e in text)
        return covered / len(entities)


# ===================== 便捷函数 =====================

# 全局单例
pre_filter = FastPreFilter()


def quick_check(
    user_query: str,
    generated_text: str,
    image_resources: List[Dict[str, Any]] = None,
) -> PreFilterResult:
    """
    快速预筛选的便捷函数。
    返回 PreFilterResult，调用方根据 result.suspicious 决定是否进入深度评估。
    """
    return pre_filter.check(user_query, generated_text, image_resources)