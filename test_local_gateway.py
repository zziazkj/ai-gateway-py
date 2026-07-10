# -*- coding: utf-8 -*-
"""
测试本地 AI Gateway - 使用本地 URL 调用
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from openai import OpenAI

# 使用本地网关地址
client = OpenAI(
    api_key="test-key",  # 网关会使用自己配置的 key
    base_url="http://localhost:8080/v1"
)

print("=" * 50)
print("测试本地 AI Gateway")
print("=" * 50)

# 测试 1: 基本调用
print("\n[测试 1] 基本调用 glm-4-flash:")
try:
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": "你好，用一句话介绍你自己"}]
    )
    print(f"回复: {response.choices[0].message.content}")
    print(f"模型: {response.model}")
    print(f"Token 用量: {response.usage}")
    print("[OK] 测试通过")
except Exception as e:
    print(f"[FAIL] 测试失败: {e}")

# 测试 2: 再次调用相同内容（测试缓存命中）
print("\n[测试 2] 再次调用相同内容（测试缓存）:")
try:
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": "你好，用一句话介绍你自己"}]
    )
    print(f"回复: {response.choices[0].message.content}")
    print("[OK] 测试通过（应该命中缓存）")
except Exception as e:
    print(f"[FAIL] 测试失败: {e}")

# 测试 3: 流式调用
print("\n[测试 3] 流式调用:")
try:
    stream = client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": "说三个笑话"}],
        stream=True
    )
    print("回复: ", end="")
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()
    print("[OK] 测试通过")
except Exception as e:
    print(f"[FAIL] 测试失败: {e}")

print("\n" + "=" * 50)
print("测试完成！")
print("=" * 50)
