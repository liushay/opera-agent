from last.embedding_raw import get_text_embedding
from last.similarity_calc import calc_cosine

# 模拟知识库文本库
knowledge_base = [
    "RAG全称检索增强生成，用于解决大模型知识过时问题",
    "LangGraph用于构建循环式多步骤Agent智能体",
    "向量数据库用于存储文本Embedding，快速语义检索",
    "Python uv是新一代包管理工具，比pip更快"
]

def search_knowledge(query: str, top_k: int = 2):
    query_vec = get_text_embedding(query)
    score_list = []
    # 遍历库内所有文本计算相似度
    for text in knowledge_base:
        text_vec = get_text_embedding(text)
        score = calc_cosine(query_vec, text_vec)
        score_list.append((score, text))
    # 按相似度从高到低排序，取前top_k
    score_list.sort(reverse=True, key=lambda x: x[0])
    return score_list[:top_k]

if __name__ == "__main__":
    user_query = "怎么解决大模型知识滞后"
    result = search_knowledge(user_query, top_k=2)
    print("检索结果：")
    for score, text in result:
        print(f"相似度{score}：{text}")