# tests/test_reranker.py Reranker 精排模块测试
# 运行：python -m tests.test_reranker
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from rag.reranker import reranker


def test_reranker():
    print("=" * 50)
    print("测试 LLMReranker 精排")
    print("=" * 50)

    # 构造测试文档：一个相关，两个无关
    docs = [
        Document(page_content="京剧脸谱的色彩具有强烈的象征意义，红色代表忠勇正义，黑色代表刚正不阿。"),
        Document(page_content="Spring Boot 是一个用于简化 Java 应用开发的框架。"),
        Document(page_content="量子计算利用量子比特进行信息处理，具有叠加和纠缠特性。"),
    ]

    # 关闭 LLM 精排（测试关键词降级逻辑，避免依赖 Ollama）
    test_reranker = reranker
    test_reranker.enable = False

    # 精排（此时走关键词预排）
    result = test_reranker.rerank("京剧脸谱的色彩象征意义是什么", docs, top_k=2)
    assert len(result) > 0, "精排结果不应为空"
    assert "京剧脸谱" in result[0].page_content, "最相关的脸谱文档应排在第一位"
    print(f"  [PASS] 精排后第一名为：{result[0].page_content[:30]}...")

    # 恢复 enable
    test_reranker.enable = True

    # 测试 _tokenize
    tokens = test_reranker._tokenize("京剧脸谱Chroma")
    assert "京剧" in tokens or "脸谱" in tokens, "中文分词应包含二元组"
    assert "chroma" in tokens, "英文应转小写整体"
    print(f"  [PASS] 分词结果：{tokens}")

    # 测试 _parse_score
    assert test_reranker._parse_score("8") == 8.0
    assert test_reranker._parse_score("得分：9分") == 9.0
    assert test_reranker._parse_score("无") == 0.0
    print("  [PASS] 分数解析")

    print("=" * 50)
    print("test_reranker.py 全部通过！")
    print("=" * 50)


if __name__ == "__main__":
    test_reranker()