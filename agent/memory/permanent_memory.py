# agent/memory/permanent_memory.py 永久静态记忆模块
# 功能：
#   1. 存放系统规则、领域规范、技术文档、用户固定偏好
#   2. 全局共享，极少修改
#   3. 采用独立存储空间（与任务级、会话时序记忆完全隔离）
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from utils.logger import log_info, log_warn, log_error
from utils.rag_exceptions import BaseRAGException


class PermanentMemoryException(BaseRAGException):
    """永久静态记忆异常"""
    code = 5101
    msg = "永久静态记忆操作失败"


class PermanentMemory:
    """
    永久静态记忆管理器
    存储媒介：本地JSON文件（独立于向量库，实现分层隔离）
    数据分类：
      - system_rules: 系统规则
      - domain_spec:  领域规范
      - tech_docs:    技术文档
      - user_prefs:   用户固定偏好
    """

    # 记忆类别
    CATEGORIES = ("system_rules", "domain_spec", "tech_docs", "user_prefs")

    def __init__(self, storage_path: str = "./memory_store/permanent_memory.json"):
        self.storage_path = storage_path
        self._storage: Dict[str, List[Dict[str, Any]]] = {}
        self._load()

    # ===================== 存储读写 =====================

    def _load(self):
        """从磁盘加载全部永久记忆"""
        import os
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            if os.path.exists(self.storage_path):
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for cat in self.CATEGORIES:
                    self._storage[cat] = data.get(cat, [])
            else:
                for cat in self.CATEGORIES:
                    self._storage[cat] = []
                self._save()
            log_info("永久静态记忆", f"加载完成，路径={self.storage_path}")
        except Exception as e:
            log_error("永久静态记忆加载失败", f"路径={self.storage_path}", e)
            # 加载失败不阻断，初始化为空
            for cat in self.CATEGORIES:
                self._storage[cat] = []

    def _save(self):
        """持久化到磁盘"""
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self._storage, f, ensure_ascii=False, indent=2)
            log_info("永久静态记忆", "保存成功")
        except Exception as e:
            err = PermanentMemoryException("永久静态记忆保存失败", e)
            log_error("永久静态记忆保存失败", self.storage_path, e)
            raise err

    # ===================== 增删改查 =====================

    def add(
        self,
        category: str,
        content: str,
        title: str = "",
        tags: Optional[List[str]] = None,
        source: str = "",
    ) -> str:
        """
        新增一条永久记忆
        category: system_rules / domain_spec / tech_docs / user_prefs
        content: 记忆内容
        title: 标题（可选）
        tags: 标签列表（可选）
        source: 来源（可选）
        """
        if category not in self.CATEGORIES:
            raise PermanentMemoryException(f"非法记忆类别：{category}，可选：{self.CATEGORIES}")

        mem_id = str(uuid.uuid4())
        item = {
            "id": mem_id,
            "category": category,
            "title": title,
            "content": content,
            "tags": tags or [],
            "source": source,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._storage[category].append(item)
        self._save()
        log_info("永久静态记忆", f"新增[{category}]记忆，id={mem_id}")
        return mem_id

    def update(self, mem_id: str, **kwargs) -> bool:
        """更新指定永久记忆的字段（content/title/tags等）"""
        for cat in self.CATEGORIES:
            for item in self._storage[cat]:
                if item["id"] == mem_id:
                    allow_fields = {"content", "title", "tags", "source", "category"}
                    for k, v in kwargs.items():
                        if k in allow_fields:
                            if k == "category" and v not in self.CATEGORIES:
                                raise PermanentMemoryException(f"非法类别：{v}")
                            item[k] = v
                    item["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # 如果修改了类别，需要搬到新类别的列表
                    if "category" in kwargs and kwargs["category"] != cat:
                        new_cat = kwargs["category"]
                        self._storage[cat].remove(item)
                        self._storage[new_cat].append(item)
                    self._save()
                    log_info("永久静态记忆", f"更新记忆 id={mem_id}")
                    return True
        log_warn("永久静态记忆", f"未找到记忆 id={mem_id}")
        return False

    def delete(self, mem_id: str) -> bool:
        """删除指定永久记忆"""
        for cat in self.CATEGORIES:
            for item in self._storage[cat]:
                if item["id"] == mem_id:
                    self._storage[cat].remove(item)
                    self._save()
                    log_info("永久静态记忆", f"删除记忆 id={mem_id}")
                    return True
        log_warn("永久静态记忆", f"未找到记忆 id={mem_id}")
        return False

    def get(self, mem_id: str) -> Optional[Dict[str, Any]]:
        """按ID获取一条永久记忆"""
        for cat in self.CATEGORIES:
            for item in self._storage[cat]:
                if item["id"] == mem_id:
                    return item
        return None

    def query(
        self,
        category: Optional[str] = None,
        keyword: str = "",
    ) -> List[Dict[str, Any]]:
        """
        查询永久记忆
        category: 按类别过滤（None为全部）
        keyword: 按关键词在标题/内容/标签中搜索
        """
        result = []
        cats = [category] if category else list(self.CATEGORIES)
        for cat in cats:
            if cat not in self.CATEGORIES:
                continue
            for item in self._storage[cat]:
                if keyword:
                    haystack = f"{item['title']} {item['content']} {' '.join(item['tags'])}"
                    if keyword.lower() not in haystack.lower():
                        continue
                result.append(item)
        return result

    def _tokenize(self, text: str) -> List[str]:
        """
        对查询/文本进行多粒度分词：
        1. 连续英文/数字为一个token
        2. 中文按单字和相邻二元组拆分（提升短查询命中率）
        """
        import re
        tokens = set()
        # 英文/数字整体
        for m in re.findall(r"[a-zA-Z0-9]+", text.lower()):
            tokens.add(m.lower())
        # 中文部分：单字 + 相邻二元组
        cn_chars = re.findall(r"[\u4e00-\u9fff]", text)
        if cn_chars:
            tokens.update(cn_chars)
            for i in range(len(cn_chars) - 1):
                tokens.add(cn_chars[i] + cn_chars[i + 1])
        return list(tokens)

    def search_for_agent(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        供Agent使用的检索接口：基于关键词匹配返回最相关的永久记忆
        不依赖向量库，使用关键词重叠度打分（轻量、快速、可靠）
        """
        query_terms = set(self._tokenize(query))
        if not query_terms:
            return []

        scored_items = []
        for cat in self.CATEGORIES:
            for item in self._storage[cat]:
                haystack = f"{item['title']} {item['content']} {' '.join(item['tags'])}".lower()
                # 对存储内容也进行分词
                hay_terms = set(self._tokenize(haystack))
                hit_terms = query_terms & hay_terms
                if hit_terms:
                    # 分数 = 命中词数 / 查询词数 + 中文二元组加权 + 标题命中加权
                    score = len(hit_terms) / len(query_terms)
                    # 中文二元组命中率更高，给予加权
                    bi_hits = sum(1 for t in hit_terms if len(t) == 2 and any('\u4e00' <= c <= '\u9fff' for c in t))
                    score += bi_hits * 0.05
                    if item["title"] and any(t in item["title"].lower() for t in hit_terms):
                        score += 0.1
                    scored_items.append((score, item))

        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_items[:top_k]]

    def get_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """返回全部永久记忆（按类别分组）"""
        return {cat: list(items) for cat, items in self._storage.items()}

    def count(self) -> int:
        """返回永久记忆总条数"""
        return sum(len(items) for items in self._storage.values())

    # ===================== 系统内置规则注入 =====================

    def seed_default_rules(self):
        """注入默认系统规则与领域规范（幂等，仅当对应类别为空时注入）"""
        if not self._storage["system_rules"]:
            self.add(
                category="system_rules",
                title="检索质量规范",
                content=(
                    "知识库检索要求：1.必须优先返回与用户问题最相关的文档片段；"
                    "2.禁止返回无关或低相关性内容；3.同一文档不重复返回；"
                    "4.检索结果需覆盖用户问题中的关键概念。"
                ),
                tags=["检索", "规范", "RAG"],
                source="system_builtin",
            )
        if not self._storage["domain_spec"]:
            self.add(
                category="domain_spec",
                title="RAG领域知识",
                content=(
                    "RAG（Retrieval-Augmented Generation）检索增强生成，核心流程："
                    "文档加载→文本分割→Embedding向量化→向量库存储→检索→大模型问答。"
                    "常见向量库包括Chroma、Milvus、FAISS、Redis Vector。"
                    "BM25是关键词检索算法，适合精确词匹配。"
                ),
                tags=["RAG", "向量库", "BM25", "分块"],
                source="system_builtin",
            )
            self.add(
                category="domain_spec",
                title="LangGraph智能体",
                content=(
                    "LangGraph是LangChain生态的智能体编排框架，用于实现具备循环、分支、"
                    "反思、多轮迭代逻辑的复杂Agent流程。利用图结构（StateGraph）组织节点"
                    "和边，支持条件路由和状态管理。"
                ),
                tags=["LangGraph", "Agent", "图", "状态"],
                source="system_builtin",
            )
        if not self._storage["tech_docs"]:
            self.add(
                category="tech_docs",
                title="Chroma向量库检索",
                content=(
                    "Chroma是本地轻量向量库，无需部署服务。使用Embedding模型将文本转为"
                    "向量存入，检索时通过相似度计算（如余弦相似度、MMR）召回最相关文档。"
                    "常用检索方式：similarity_search、MMR（最大边际相关性）。"
                ),
                tags=["Chroma", "向量", "检索", "MMR"],
                source="system_builtin",
            )
        if not self._storage["user_prefs"]:
            self.add(
                category="user_prefs",
                title="默认回答偏好",
                content="回答要求：简洁通顺，严格依据提供资料，不编造未出现的信息。",
                tags=["回答偏好"],
                source="system_builtin",
            )
        log_info("永久静态记忆", "默认规则注入完成")


# 全局单例
permanent_memory = PermanentMemory()
permanent_memory.seed_default_rules()