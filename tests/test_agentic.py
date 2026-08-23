# tests/test_agentic.py Agentic RAG 组件测试
# 运行：python -m tests.test_agentic
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_agentic_imports():
    """验证 Agentic RAG 各组件可导入"""
    from rag.agentic import query_rewriter, document_grader, answer_grader, auto_retriever
    assert query_rewriter is not None
    assert document_grader is not None
    assert answer_grader is not None
    assert auto_retriever is not None
    print("  [PASS] Agentic RAG 组件导入")


def test_answer_grader_logic():
    """验证 AnswerGrader 的判定逻辑（不依赖外部 LLM）"""
    # 重定向 _ask_llm 避免真实调用
    from rag.agentic.answer_grader import answer_grader

    # 模拟 yes
    answer_grader._ask_llm = lambda prompt: "yes"
    r1 = answer_grader.grade_faithfulness("q", "answer", "context")
    assert r1["passed"] is True and r1["score"] == 1.0
    print("  [PASS] Faithfulness yes 判定")

    # 模拟 no
    answer_grader._ask_llm = lambda prompt: "no"
    r2 = answer_grader.grade_answer_relevance("q", "answer")
    assert r2["passed"] is False and r2["score"] == 0.0
    print("  [PASS] Answer Relevance no 判定")


def test_document_grader_filter():
    """验证 DocumentGrader 的过滤逻辑"""
    from langchain_core.documents import Document
    from rag.agentic.document_grader import document_grader

    # 模拟全部相关
    document_grader._grade_single = lambda q, d, m: "yes"

    docs = [
        Document(page_content="京剧脸谱红色代表忠勇。"),
        Document(page_content="京剧四大名旦是梅兰芳等人。"),
        Document(page_content="量子计算原理。"),
    ]
    graded = document_grader.grade("京剧", docs, mode="binary")
    assert len(graded) == 3
    assert all(g["relevant"] for g in graded)
    print("  [PASS] DocumentGrader 全部相关判定")

    # 混合
    def fake_grade(q, d, m):
        return "yes" if "京剧" in d else "no"

    document_grader._grade_single = fake_grade
    filtered = document_grader.filter_relevant("京剧", docs, mode="binary")
    assert len(filtered) == 2
    print("  [PASS] DocumentGrader 过滤无关文档")


def test_context_metrics():
    """验证 Context Precision / Recall 计算"""
    from evaluation.advanced_evaluator import advanced_evaluator

    # 相关文档排在位置 1 和 3
    precision = advanced_evaluator.context_precision(
        "q", [], relevant_indices=[1, 3]
    )
    # (1/1 + 2/3) / 2 = (1 + 0.667) / 2 ≈ 0.8334
    assert abs(precision - 0.8334) < 0.01, f"Context Precision 计算错误: {precision}"
    print(f"  [PASS] Context Precision = {precision}")

    recall = advanced_evaluator.context_recall(
        "q", [], relevant_indices=[1, 3], total_relevant=4
    )
    assert abs(recall - 0.5) < 0.001
    print(f"  [PASS] Context Recall = {recall}")


def test_agent_task_success():
    """验证 Agent Task Success"""
    from evaluation.advanced_evaluator import advanced_evaluator

    assert advanced_evaluator.agent_task_success(True, True) == 1.0
    assert advanced_evaluator.agent_task_success(True, False) == 0.0
    print("  [PASS] Agent Task Success")


if __name__ == "__main__":
    print("=" * 50)
    print("测试 Agentic RAG + 高级评测")
    print("=" * 50)
    test_agentic_imports()
    test_answer_grader_logic()
    test_document_grader_filter()
    test_context_metrics()
    test_agent_task_success()
    print("=" * 50)
    print("test_agentic.py 全部通过！")
    print("=" * 50)