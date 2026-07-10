"""
测试智谱 API 是否可用
"""

import httpx
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("UPSTREAM_API_KEY", "").strip()
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

print(f"API Key: {API_KEY[:10]}...{API_KEY[-4:] if len(API_KEY) > 14 else ''}")
print(f"API Address: {BASE_URL}")
print(f"Key Length: {len(API_KEY)}")
print()

# 测试1：检查模型列表
print("=" * 50)
print("Test 1: Get Model List")
print("=" * 50)

try:
    with httpx.Client(timeout=10) as client:
        resp = client.get(
            f"{BASE_URL}/models",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            print("[OK] API Key is valid!")
            data = resp.json()
            models = [m.get("id", "") for m in data.get("data", [])]
            print(f"Available models: {models[:10]}")
        else:
            print(f"[ERROR] {resp.text[:300]}")
except Exception as e:
    print(f"[FAIL] Connection failed: {e}")

print()

# 测试2：发送聊天请求
print("=" * 50)
print("Test 2: Chat Completions (glm-4-flash)")
print("=" * 50)

try:
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{BASE_URL}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            },
            json={
                "model": "glm-4-flash",
                "messages": [{"role": "user", "content": "hi, reply with one word"}]
            }
        )
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            print(f"[OK] Success!")
            print(f"Answer: {answer}")
        else:
            print(f"[ERROR] {resp.text[:300]}")
except Exception as e:
    print(f"[FAIL] Connection failed: {e}")
