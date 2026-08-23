# 验证戏曲知识库检索功能
import sys
sys.path.insert(0, '.')
from rag.vectorstore import hybrid_retrieve, kb

def main():
    print("=" * 60)
    print("戏曲知识库检索验证")
    print("=" * 60)

    # 1. 检查知识库状态
    data = kb.vector_store.get()
    print(f"\n[1] 向量库状态")
    print(f"  文档块数: {len(data['ids'])}")

    # 2. 重建BM25索引（确认生效）
    print(f"\n[2] 重建BM25索引...")
    kb.rebuild_full_bm25()
    from rag.vectorstore import bm25_kb
    print(f"  BM25索引文本数: {len(bm25_kb.corpus_texts)}")

    # 3. 实际检索验证
    print(f"\n[3] 戏曲检索验证")
    test_queries = [
        '京剧脸谱的色彩象征意义是什么',
        '京剧四大名旦分别是谁',
        '黄梅戏天仙配讲述了什么故事',
        '昆曲牡丹亭的艺术价值',
        '川剧变脸的表演原理'
    ]
    for q in test_queries:
        docs = hybrid_retrieve(q)
        print(f"\n  查询: {q}")
        for i, d in enumerate(docs, 1):
            text = d.page_content[:60].replace('\n', ' ')
            print(f"    [{i}] {text}...")

    print("\n" + "=" * 60)
    print("戏曲知识库检索验证完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()