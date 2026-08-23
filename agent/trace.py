# agent/trace.py Agent 可观测性（Trace/Observability）
# 功能：
#   1. 记录 Agent 每次调用的完整轨迹（步骤、耗时、输入输出、工具调用、错误）
#   2. 生成结构化 Trace JSON 与 Markdown 报告
#   3. 提供汇总统计（调用次数、成功率、平均耗时、工具使用频率）
# 设计：
#   - 全局单例 tracer，线程安全（加锁）
#   - 无需改动现有 Agent 逻辑即可记录（通过 with_trace 装饰器/手动 span）
import json
import os
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional


class AgentTracer:
    """Agent 调用追踪器"""

    def __init__(self, trace_dir: str = "./agent_traces"):
        self.trace_dir = trace_dir
        os.makedirs(trace_dir, exist_ok=True)
        self._lock = threading.Lock()
        # 当前活跃 trace 栈
        self._active_traces: Dict[str, Dict] = {}
        # 汇总统计
        self._stats = {
            "total_calls": 0,
            "success_calls": 0,
            "failed_calls": 0,
            "total_duration_ms": 0,
            "tool_usage": {},
        }

    # ===================== Trace 生命周期 =====================

    def start_trace(
        self,
        name: str,
        session_id: str = "",
        user_query: str = "",
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        开启一条新 Trace
        Returns: trace_id
        """
        trace_id = str(uuid.uuid4())
        trace = {
            "trace_id": trace_id,
            "name": name,
            "session_id": session_id,
            "user_query": user_query,
            "metadata": metadata or {},
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "start_ts": time.time(),
            "end_time": None,
            "duration_ms": None,
            "status": "running",  # running / success / error
            "error": None,
            "steps": [],  # 内部步骤
            "tool_calls": [],  # 工具调用记录
            "result": None,
        }
        with self._lock:
            self._active_traces[trace_id] = trace
        return trace_id

    def add_step(
        self,
        trace_id: str,
        step_name: str,
        detail: str = "",
        metadata: Optional[Dict] = None,
    ) -> None:
        """记录一个内部步骤"""
        with self._lock:
            trace = self._active_traces.get(trace_id)
            if trace is None:
                return
            trace["steps"].append({
                "step": step_name,
                "detail": detail,
                "metadata": metadata or {},
                "ts": time.time(),
            })

    def record_tool_call(
        self,
        trace_id: str,
        tool_name: str,
        params: Any = None,
        result: Any = None,
        duration_ms: float = 0.0,
        success: bool = True,
    ) -> None:
        """记录一次工具调用"""
        with self._lock:
            trace = self._active_traces.get(trace_id)
            if trace is None:
                return
            call = {
                "tool_name": tool_name,
                "params": self._truncate(params),
                "result": self._truncate(result),
                "duration_ms": round(duration_ms, 2),
                "success": success,
                "ts": time.time(),
            }
            trace["tool_calls"].append(call)
            # 更新工具使用统计
            self._stats["tool_usage"][tool_name] = (
                self._stats["tool_usage"].get(tool_name, 0) + 1
            )

    def end_trace(
        self,
        trace_id: str,
        status: str = "success",
        result: Any = None,
        error: str = None,
    ) -> Dict[str, Any]:
        """结束一条 Trace 并保存"""
        trace = None
        with self._lock:
            trace = self._active_traces.pop(trace_id, None)
            if trace is None:
                return {}

            trace["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            trace["duration_ms"] = round((time.time() - trace["start_ts"]) * 1000, 2)
            trace["status"] = status
            trace["result"] = self._truncate(result)
            trace["error"] = error

            # 更新统计
            self._stats["total_calls"] += 1
            if status == "success":
                self._stats["success_calls"] += 1
            else:
                self._stats["failed_calls"] += 1
            self._stats["total_duration_ms"] += trace["duration_ms"]

        # 保存 Trace
        self._save_trace(trace)
        return trace

    # ===================== 查询接口 =====================

    def get_trace(self, trace_id: str) -> Optional[Dict]:
        """获取指定 Trace"""
        with self._lock:
            if trace_id in self._active_traces:
                return self._active_traces[trace_id]
        path = os.path.join(self.trace_dir, f"{trace_id}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def list_traces(self, limit: int = 20) -> List[Dict]:
        """列出最近 N 条 Trace 概要"""
        files = sorted(
            [f for f in os.listdir(self.trace_dir) if f.endswith(".json")],
            reverse=True,
        )[:limit]
        result = []
        for fname in files:
            with open(os.path.join(self.trace_dir, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            result.append({
                "trace_id": data["trace_id"],
                "name": data["name"],
                "status": data["status"],
                "duration_ms": data["duration_ms"],
                "start_time": data["start_time"],
                "user_query": data.get("user_query", ""),
                "tool_call_count": len(data.get("tool_calls", [])),
            })
        return result

    def get_stats(self) -> Dict[str, Any]:
        """获取汇总统计"""
        with self._lock:
            avg_duration = (
                round(self._stats["total_duration_ms"] / self._stats["total_calls"], 2)
                if self._stats["total_calls"]
                else 0
            )
            success_rate = (
                round(self._stats["success_calls"] / self._stats["total_calls"] * 100, 2)
                if self._stats["total_calls"]
                else 0
            )
            return {
                "total_calls": self._stats["total_calls"],
                "success_calls": self._stats["success_calls"],
                "failed_calls": self._stats["failed_calls"],
                "success_rate": success_rate,
                "avg_duration_ms": avg_duration,
                "tool_usage": self._stats["tool_usage"],
            }

    # ===================== 内部工具 =====================

    def _truncate(self, value: Any, max_len: int = 500) -> Any:
        """截断超长内容"""
        if isinstance(value, str) and len(value) > max_len:
            return value[:max_len] + "...(截断)"
        if isinstance(value, list):
            return [self._truncate(v, max_len // 2) for v in value[:10]]
        if isinstance(value, dict):
            return {k: self._truncate(v, max_len // 2) for k, v in list(value.items())[:10]}
        return value

    def _save_trace(self, trace: Dict) -> str:
        """保存 Trace 到磁盘"""
        path = os.path.join(self.trace_dir, f"{trace['trace_id']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
        return path

    # ===================== 装饰器 =====================

    def trace(self, name: str = None, session_id_field: str = None, query_field: str = None):
        """装饰器：自动记录函数调用的 Trace"""
        import functools

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                trace_name = name or func.__name__
                # 从 kwargs 中提取 session_id / query（若指定字段名）
                session_id = (
                    str(kwargs.get(session_id_field, "")) if session_id_field else ""
                )
                user_query = str(kwargs.get(query_field, "")) if query_field else ""

                trace_id = self.start_trace(
                    name=trace_name,
                    session_id=session_id,
                    user_query=user_query,
                )
                try:
                    result = func(*args, **kwargs)
                    self.end_trace(trace_id, status="success", result=result)
                    return result
                except Exception as e:
                    self.end_trace(trace_id, status="error", error=str(e))
                    raise

            return wrapper

        return decorator


# 全局单例
agent_tracer = AgentTracer()

# 兼容名称
tracer = agent_tracer