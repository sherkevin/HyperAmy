#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 Embedding API"""
import os
from dotenv import load_dotenv
import requests

load_dotenv()

api_key = os.getenv("API_KEY")
url = os.getenv("API_URL_EMBEDDINGS", "https://llmapi.paratera.com/v1/embeddings")

print(f"测试 Embedding API")
print(f"URL: {url}")
print(f"模型: GLM-Embedding-3")
print()

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": "GLM-Embedding-3",
    "input": ["test embedding"]
}

try:
    response = requests.post(url, headers=headers, json=payload, timeout=10)
    print(f"状态码: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 成功!")
        print(f"响应键: {list(result.keys())}")
        if "data" in result and len(result["data"]) > 0:
            print(f"Embedding 维度: {len(result['data'][0]['embedding'])}")
    else:
        print(f"❌ 失败: {response.text[:500]}")
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()


