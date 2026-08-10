import requests
import json

def get_text_embedding(text: str, model: str = "nomic-embed-text") -> list[float]:
    url = "http://localhost:11434/api/embeddings"
    payload = {
        "model": model,
        "prompt": text
    }
    resp = requests.post(url, json=payload)
    data = json.loads(resp.text)
    # 返回文本对应的向量数组
    return data["embedding"]

if __name__ == "__main__":
    text1 = "Agent智能体可以调用工具完成任务"
    vec1 = get_text_embedding(text1)
    print(f"向量长度：{len(vec1)}")
    print(f"向量前10位数值：{vec1[:10]}")