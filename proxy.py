"""
上游代理：与 LLM API 通信，支持主备切换 + 重试
"""

import asyncio
import logging
import time
from typing import Optional

import httpx

from config import AppConfig
from circuit_breaker import CircuitBreaker

log = logging.getLogger("proxy")


class UpstreamProvider:
    """
    上游 LLM 提供商代理

    功能：
    - 主备切换：主 provider 失败自动切换到备用
    - 请求重试：指数退避重试
    - 熔断保护：连续失败后自动切断
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.base_url = cfg.upstream.base_url
        self.timeout = cfg.upstream.timeout_seconds
        self.max_retries = cfg.upstream.max_retries

        # 熔断器
        self.breaker = CircuitBreaker(
            failure_threshold=cfg.circuit_breaker.failure_threshold,
            recovery_timeout=cfg.circuit_breaker.recovery_timeout_seconds,
        ) if cfg.circuit_breaker.enabled else None

        # HTTP 客户端
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))

    async def call(self, body: dict, user_api_key: str = "") -> dict:
        """
        调用上游 LLM

        优先使用用户提供的 API Key，
        若无则使用服务器配置的 Key
        """
        api_key = (user_api_key or self.cfg.upstream_api_key).strip()
        if not api_key:
            log.error("❌ 无可用的 API Key")
            return {"error": "No API key available"}

        # 去掉可能存在的 Bearer 前缀，避免重复
        while api_key.lower().startswith("bearer "):
            api_key = api_key[7:].strip()

        log.info(f"🔑 最终使用的 Key: 前10位={api_key[:10]}, 长度={len(api_key)}")

        # 构建请求 URL
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        # 带熔断器的调用
        if self.breaker:
            result = await self.breaker.execute(
                lambda: self._do_request_with_retry(url, headers, body)
            )
            if result is not None:
                return result
            return {"error": "Upstream provider unavailable (circuit open)"}

        # 无熔断器：直接重试
        result = await self._do_request_with_retry(url, headers, body)
        return result or {"error": "All upstream providers unavailable"}

    async def _do_request_with_retry(
        self, url: str, headers: dict, body: dict
    ) -> Optional[dict]:
        """带指数退避的重试"""
        last_error = None

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                # 指数退避：500ms, 1000ms, 2000ms...
                wait_time = 0.5 * (2 ** (attempt - 1))
                log.info(f"🔄 重试 {attempt}/{self.max_retries}, 等待 {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

            try:
                result = await self._do_request(url, headers, body)
                if result is not None and "error" not in result:
                    return result
                last_error = result
            except Exception as e:
                last_error = {"error": str(e)}
                log.warning(f"⚠️  请求失败 (attempt {attempt + 1}): {e}")

        return last_error

    async def _do_request(
        self, url: str, headers: dict, body: dict
    ) -> Optional[dict]:
        """发送实际的 HTTP 请求"""
        try:
            import time as _time
            t0 = _time.time()
            log.info(f"📡 调用上游: {url}")
            auth = headers.get('Authorization', '')
            log.info(f"🔑 Authorization: {auth[:30]}")
            
            # 检查是否为流式请求
            is_stream = body.get("stream", False)
            
            response = await self.client.post(url, json=body, headers=headers)
            elapsed = int((_time.time() - t0) * 1000)
            log.info(f"📡 上游响应: {response.status_code} | 耗时: {elapsed}ms")
            
            if response.status_code != 200:
                log.error(f"❌ 上游返回错误: {response.status_code} - {response.text[:200]}")
            response.raise_for_status()
            
            # 流式响应：读取所有内容并拼接
            if is_stream:
                log.info("📡 流式响应，拼接结果...")
                full_content = ""
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            import json
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_content += content
                        except:
                            pass
                
                # 构造标准响应格式
                return {
                    "id": f"stream-{int(t0)}",
                    "object": "chat.completion",
                    "created": int(t0),
                    "model": body.get("model", ""),
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": full_content},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                }
            
            # 非流式响应：直接解析 JSON
            return response.json()

        except httpx.TimeoutException:
            log.error(f"⏰ 请求超时: {url}")
            return None

        except httpx.HTTPStatusError as e:
            log.error(f"❌ HTTP 错误 {e.response.status_code}: {e.response.text[:200]}")
            return {"error": f"HTTP {e.response.status_code}"}

        except Exception as e:
            log.error(f"❌ 请求异常: {e}")
            return None

    async def close(self):
        """关闭 HTTP 客户端"""
        await self.client.aclose()
