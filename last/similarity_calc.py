import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from last.embedding_raw import get_text_embedding

def calc_cosine(vec_a: list, vec_b: list) -> float:
    # 转为numpy二维数组适配库函数
    arr_a = np.array(vec_a).reshape(1, -1)
    arr_b = np.array(vec_b).reshape(1, -1)
    sim = cosine_similarity(arr_a, arr_b)
    return round(float(sim[0][0]), 4)

if __name__ == "__main__":
    # 三组文本对比测试
    target = "大模型RAG检索增强技术"
    text_a = "RAG通过知识库检索辅助大模型回答问题"
    text_b = "今天中午吃了牛肉面"

    vec_target = get_text_embedding(target)
    vec_a = get_text_embedding(text_a)
    vec_b = get_text_embedding(text_b)

    sim1 = calc_cosine(vec_target, vec_a)
    sim2 = calc_cosine(vec_target, vec_b)
    print(f"目标文本与RAG相关文本相似度：{sim1}")
    print(f"目标文本与无关文本相似度：{sim2}")