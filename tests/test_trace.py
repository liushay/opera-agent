# tests/test_trace.py Agent Trace/Observability 测试
# 运行：python -m tests.test_trace
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.trace import agent_tracer


def test_trace_lifecycle():
    """验证 Trace 生命周期：start → step → tool_call → end → stats"""
    # 开启 Trace
    tid = agent_tracer.start_trace(
        name="test_agent",
        session_id="sess_001",
        user_query="京剧四大名旦",
        metadata={"model": "qwen2:7b"},
    )
    assert tid, "trace_id 不应为空"

    # 记录步骤
    agent_tracer.add_step(tid, "plan", "规划工具调用")
    agent_tracer.add_step(tid, "reflect", "反思校验")

    # 记录工具调用（成功）
    agent_tracer.record_tool_call(
        tid, "kb_search", {"query": "京剧四大名旦"},
        "检索到4条文档", 235.5, True,
    )

    # 结束 Trace
    trace = agent_tracer.end_trace(tid, status="success", result={"answer": "ok"})
    assert trace["status"] == "success"
    assert trace["duration_ms"] is not None
    assert len(trace["steps"]) == 2
    assert len(trace["tool_calls"]) == 1
    assert trace["tool_calls"][0]["tool_name"] == "kb_search"
    print("  [PASS] Trace 生命周期")

    # 获取已保存的 Trace
    saved = agent_tracer.get_trace(tid)
    assert saved is not None, "Trace 应能按 id 查询"
    print("  [PASS] Trace 查询")


def test_trace_stats():
    """验证统计功能"""
    before = agent_tracer.get_stats()
    assert "total_calls" in before
    assert "tool_usage" in before
    print("  [PASS] Trace 统计接口")


def test_trace_decorator():
    """验证装饰器用法"""
    @agent_tracer.trace(name="decorated", session_id_field="session_id", query_field="query")
    def fake_agent(session_id="", query=""):
        return "answer_ok"

    result = fake_agent(session_id="s2", query="黄梅戏天仙配")
    assert result == "answer_ok"
    stats = agent_tracer.get_stats()
    assert stats["total_calls"] >= 1
    print("  [PASS] Trace 装饰器")


def test_trace_error():
    """验证异常轨迹记录"""
    tid = agent_tracer.start_trace("error_test", user_query="bad")
    agent_tracer.add_step(tid, "step", "do something")
    # 不结束（模拟运行中异常）也应有 active 记录；此处正常结束为 error
    trace = agent_tracer.end_trace(tid, status="error", error="模拟错误")
    assert trace["status"] == "error"
    assert trace["error"] == "模拟错误"
    print("  [PASS] Trace 错误轨迹")


if __name__ == "__main__":
    print("=" * 50)
    print("测试 Agent Trace/Observability")
    print("=" * 50)
    test_trace_lifecycle()
    test_trace_stats()
    test_trace_decorator()
    test_trace_error()
    print("=" * 50)
    print("test_trace.py 全部通过！")
    print("=" * 50)