import requests
import json

def ollama_chat(prompt: str, model: str = "qwen2:7b") -> str:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    resp = requests.post(url, json=payload)
    result = json.loads(resp.text)
    return result["response"]

if __name__ == "__main__":
    res = ollama_chat("简单介绍什么是Agent智能体")
    print(res)