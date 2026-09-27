# agent/memory/task_memory.py 任务级记忆模块
# 功能：
#   1. 每个任务创建独立记忆空间，保存任务需求、目标、中间产物、约束
#   2. 任务数据互相隔离，支持父子任务嵌套
#   3. 采用独立存储空间（与永久静态、会话时序记忆完全隔离）
import json
import os
import uuid
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional

from utils.logger import log_info, log_warn, log_error
from utils.rag_exceptions import BaseRAGException


class TaskMemoryException(BaseRAGException):
    """任务级记忆异常"""
    code = 5102
    msg = "任务级记忆操作失败"


class TaskMemory:
    """
    任务级记忆管理器
    存储媒介：本地JSON目录（tasks/xxx.json），与向量库和永久记忆完全隔离
    结构：
      memory_store/tasks/
        {task_id}.json
    每个任务独立文件，任务间数据完全隔离，支持父子任务嵌套
    """

    def __init__(self, storage_dir: str = "./memory_store/tasks"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    # ===================== 路径工具 =====================

    def _task_path(self, task_id: str) -> str:
        """获取任务文件路径，校验task_id合法性"""
        task_id = str(task_id)
        # 防止路径穿越
        if task_id in ("", ".", "..") or "/" in task_id or "\\" in task_id:
            raise TaskMemoryException(f"非法任务ID：{task_id}")
        return os.path.join(self.storage_dir, f"{task_id}.json")

    def _load_task(self, task_id: str) -> Dict[str, Any]:
        """加载单个任务数据"""
        path = self._task_path(task_id)
        if not os.path.exists(path):
            raise TaskMemoryException(f"任务不存在：{task_id}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            err = TaskMemoryException(f"任务数据加载失败：{task_id}", e)
            log_error("任务级记忆", err.msg, e)
            raise err

    def _save_task(self, task_id: str, data: Dict[str, Any]):
        """保存单个任务数据"""
        path = self._task_path(task_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            log_info("任务级记忆", f"任务[{task_id}]保存成功")
        except Exception as e:
            err = TaskMemoryException(f"任务数据保存失败：{task_id}", e)
            log_error("任务级记忆", err.msg, e)
            raise err

    # ===================== 任务CRUD =====================

    def create_task(
        self,
        task_id: Optional[str] = None,
        title: str = "",
        description: str = "",
        requirements: str = "",
        objectives: List[str] = None,
        constraints: List[str] = None,
        parent_task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        创建新任务记忆空间
        task_id: 自定义任务ID（默认自动生成UUID）
        title: 任务标题
        description: 任务描述
        requirements: 任务需求
        objectives: 任务目标列表
        constraints: 任务约束列表
        parent_task_id: 父任务ID（支持父子任务嵌套）
        metadata: 附加元数据
        """
        if task_id is None:
            task_id = str(uuid.uuid4())
        task_id = str(task_id)

        # 校验父任务存在
        if parent_task_id and not os.path.exists(self._task_path(parent_task_id)):
            raise TaskMemoryException(f"父任务不存在：{parent_task_id}")

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        task_data = {
            "id": task_id,
            "title": title,
            "description": description,
            "requirements": requirements,
            "objectives": objectives or [],
            "constraints": constraints or [],
            "intermediate_results": [],
            "status": "created",
            "parent_task_id": parent_task_id,
            "sub_task_ids": [],
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now,
            "completed_at": None,
        }

        # 如果有父任务，更新父任务的 sub_task_ids
        if parent_task_id:
            try:
                parent = self._load_task(parent_task_id)
            except TaskMemoryException:
                raise TaskMemoryException(f"父任务不存在：{parent_task_id}")
            if task_id not in parent["sub_task_ids"]:
                parent["sub_task_ids"].append(task_id)
                parent["updated_at"] = now
                self._save_task(parent_task_id, parent)

        self._save_task(task_id, task_data)
        log_info("任务级记忆", f"创建任务[{task_id}]，父任务={parent_task_id or '无'}")
        return task_data

    def get_task(self, task_id: str) -> Dict[str, Any]:
        """获取任务完整数据"""
        return self._load_task(task_id)

    def update_task(self, task_id: str, **kwargs) -> Dict[str, Any]:
        """
        更新任务信息
        可更新字段：title/description/requirements/objectives/constraints/status/metadata
        """
        task = self._load_task(task_id)
        allow_fields = {
            "title", "description", "requirements",
            "objectives", "constraints", "status", "metadata",
        }
        for k, v in kwargs.items():
            if k in allow_fields:
                task[k] = v
        task["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_task(task_id, task)
        log_info("任务级记忆", f"更新任务[{task_id}]")
        return task

    def delete_task(self, task_id: str, recursive: bool = False) -> bool:
        """
        删除任务
        recursive=True 时级联删除所有子任务
        """
        task = self._load_task(task_id)  # 确认存在
        sub_ids = task.get("sub_task_ids", [])

        if recursive:
            # 递归删除子任务
            for sub_id in sub_ids:
                self.delete_task(sub_id, recursive=True)
        elif sub_ids:
            raise TaskMemoryException(
                f"任务[{task_id}]存在{sub_ids}个子任务，需设置recursive=True级联删除"
            )

        # 从父任务中移除
        parent_id = task.get("parent_task_id")
        if parent_id:
            try:
                parent = self._load_task(parent_id)
                if task_id in parent["sub_task_ids"]:
                    parent["sub_task_ids"].remove(task_id)
                    parent["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self._save_task(parent_id, parent)
            except TaskMemoryException:
                log_warn("任务级记忆", f"父任务{parent_id}不存在，跳过父任务更新")

        path = self._task_path(task_id)
        os.remove(path)
        log_info("任务级记忆", f"删除任务[{task_id}]")
        return True

    # ===================== 任务数据操作 =====================

    def add_intermediate_result(self, task_id: str, result: Dict[str, Any]) -> str:
        """
        添加任务中间产物
        result: {type: "text/table/code/url", content: "...", note: "..."} 
        """
        task = self._load_task(task_id)
        result_id = str(uuid.uuid4())
        item = {
            "id": result_id,
            **result,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        task["intermediate_results"].append(item)
        task["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_task(task_id, task)
        return result_id

    def get_intermediate_results(self, task_id: str) -> List[Dict[str, Any]]:
        """获取任务的全部中间产物"""
        task = self._load_task(task_id)
        return task["intermediate_results"]

    def mark_completed(self, task_id: str) -> Dict[str, Any]:
        """标记任务完成"""
        task = self._load_task(task_id)
        task["status"] = "completed"
        task["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        task["updated_at"] = task["completed_at"]
        self._save_task(task_id, task)
        log_info("任务级记忆", f"任务[{task_id}]标记完成")
        return task

    # ===================== 任务查询 =====================

    def list_tasks(self, parent_task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出任务列表，可按父任务过滤"""
        tasks = []
        for fname in os.listdir(self.storage_dir):
            if not fname.endswith(".json"):
                continue
            task_id = fname[:-5]
            try:
                task = self._load_task(task_id)
                if parent_task_id is None or task.get("parent_task_id") == parent_task_id:
                    tasks.append(task)
            except TaskMemoryException:
                continue
        tasks.sort(key=lambda t: t.get("created_at", ""), reverse=True)
        return tasks

    def get_all_task_ids(self) -> List[str]:
        """获取全部任务ID"""
        return [f[:-5] for f in os.listdir(self.storage_dir) if f.endswith(".json")]

    def count(self) -> int:
        """任务总数"""
        return len(self.get_all_task_ids())

    # ===================== Agent检索接口 =====================

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

    def search_for_agent(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        供Agent使用的任务记忆检索接口：基于关键词匹配返回最相关任务
        返回任务的精简摘要信息
        """
        query_terms = set(self._tokenize(query))
        if not query_terms:
            return []

        scored_tasks = []
        for task_id in self.get_all_task_ids():
            try:
                task = self._load_task(task_id)
            except TaskMemoryException:
                continue
            haystack = " ".join([
                task.get("title", ""),
                task.get("description", ""),
                task.get("requirements", ""),
                " ".join(task.get("objectives", [])),
                " ".join(task.get("constraints", [])),
            ]).lower()
            hay_terms = set(self._tokenize(haystack))
            hit_terms = query_terms & hay_terms
            if hit_terms:
                score = len(hit_terms) / len(query_terms)
                # 中文二元组命中加权
                bi_hits = sum(1 for t in hit_terms if len(t) == 2 and any('\u4e00' <= c <= '\u9fff' for c in t))
                score += bi_hits * 0.05
                scored_tasks.append((score, task))

        scored_tasks.sort(key=lambda x: x[0], reverse=True)
        result = []
        for _, task in scored_tasks[:top_k]:
            result.append({
                "id": task["id"],
                "title": task.get("title", ""),
                "description": task.get("description", ""),
                "status": task.get("status", ""),
                "objectives": task.get("objectives", [])[:3],
                "constraints": task.get("constraints", [])[:3],
                "intermediate_results": task.get("intermediate_results", [])[-3:],
                "parent_task_id": task.get("parent_task_id"),
                "sub_task_ids": task.get("sub_task_ids", []),
                "updated_at": task.get("updated_at", ""),
            })
        return result


# 全局单例
task_memory = TaskMemory()