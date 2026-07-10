"""
语义缓存引擎 — 四层匹配策略
① 去重 → ② 精确哈希 → ③ 向量语义 → ④ Jaccard 词重叠
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from embedding import VectorEmbeddingEngine, cosine_similarity, jaccard_similarity, clean_text
from config import AppConfig

log = logging.getLogger("cache")


class CacheEntry:
    """缓存条目"""
    __slots__ = ("prompt", "response", "vector", "timestamp")

    def __init__(self, prompt: str, response: dict, vector: np.ndarray, timestamp: float):
        self.prompt = prompt
        self.response = response
        self.vector = vector
        self.timestamp = timestamp


class SemanticCache:
    """
    语义缓存：支持 Redis 持久化 + 内存降级

    缓存查找顺序：
    1. 去重检查（并发相同请求等待第一个结果）
    2. 精确哈希匹配（SHA256 完全一致）
    3. 向量语义匹配（余弦相似度 ≥ 阈值）
    4. Jaccard 词重叠匹配（兜底）
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.embedder = VectorEmbeddingEngine(dimension=cfg.cache.vector_dimension)

        # 内存缓存：{tenant_id: [CacheEntry, ...]}
        self.local_index: Dict[str, List[CacheEntry]] = {}
        self.max_per_tenant = 10000

        # 去重：等待中的并发请求
        self.pending_requests: Dict[str, asyncio.Event] = {}
        self._pending_results: Dict[str, Any] = {}

        # Redis 客户端（可选）
        self.redis = None
        if cfg.cache_mode == "redis":
            try:
                import redis.asyncio as aioredis
                self.redis = aioredis.from_url(cfg.cache.redis_url, decode_responses=True)
                log.info("✅ Redis 缓存后端已连接")
            except Exception as e:
                log.warning(f"⚠️  Redis 连接失败，降级为内存缓存: {e}")

    async def close(self):
        """关闭连接"""
        if self.redis:
            await self.redis.close()

    # ==================== 核心：缓存查找 ====================
    async def search(
        self, tenant_id: str, prompt: str, threshold: float
    ) -> Tuple[Optional[dict], float, bool]:
        """
        四层缓存查找

        Returns:
            (cached_response, similarity_score, is_hit)
        """
        import time as _t
        t_start = _t.time()
        clean_prompt = clean_text(prompt)

        # ① 去重检查
        if self.cfg.dedup.enabled:
            t0 = _t.time()
            result = await self._check_dedup(tenant_id, clean_prompt)
            ms = int((_t.time() - t0) * 1000)
            if result is not None:
                log.info(f"🔍 [去重] HIT | {ms}ms")
                return result, 1.0, True
            log.info(f"🔍 [去重] MISS | {ms}ms")

        # ② 精确哈希匹配（先查本用户，再查共享）
        if self.cfg.cache.exact_hash_enabled:
            t0 = _t.time()
            prompt_hash = hashlib.sha256(clean_prompt.encode()).hexdigest()
            # 先查本用户缓存
            result = await self._exact_lookup(tenant_id, prompt_hash)
            if result is None:
                # 再查共享缓存（"_shared" 用户）
                result = await self._exact_lookup("_shared", prompt_hash)
            ms = int((_t.time() - t0) * 1000)
            if result is not None:
                log.info(f"🔍 [精确哈希] HIT | {ms}ms")
                return result, 1.0, True
            log.info(f"🔍 [精确哈希] MISS | {ms}ms")

        # ③ 向量语义匹配
        if self.cfg.cache.vector_enabled:
            t0 = _t.time()
            query_vector = await self.embedder.embed(clean_prompt)
            ms_embed = int((_t.time() - t0) * 1000)

            # 精确哈希也存一份，用于下次精确匹配
            exact_hash = hashlib.sha256(clean_prompt.encode()).hexdigest()

            # 先查 Redis
            if self.redis:
                t0 = _t.time()
                result, score = await self._vector_search_redis(tenant_id, query_vector, threshold)
                ms = int((_t.time() - t0) * 1000)
                if result is not None:
                    log.info(f"🔍 [向量/Redis] HIT | score={score:.4f} | {ms}ms (embed={ms_embed}ms)")
                    return result, score, True
                log.info(f"🔍 [向量/Redis] MISS | {ms}ms")

            # 再查内存（先查本用户，再查共享）
            t0 = _t.time()
            result, score = self._vector_search_local(tenant_id, query_vector, threshold)
            if result is None:
                result, score = self._vector_search_local("_shared", query_vector, threshold)
            ms = int((_t.time() - t0) * 1000)
            if result is not None:
                log.info(f"🔍 [向量/本地] HIT | score={score:.4f} | {ms}ms (embed={ms_embed}ms)")
                return result, score, True
            # 显示最高分但未达阈值的条目
            entries = self.local_index.get(tenant_id, [])
            if entries:
                all_scores = []
                for e in entries:
                    if e.vector is not None:
                        s = cosine_similarity(query_vector, e.vector)
                        all_scores.append((s, e.prompt[:20]))
                all_scores.sort(reverse=True)
                if all_scores:
                    log.info(f"🔍 [向量/本地] MISS | 最高相似度={all_scores[0][0]:.4f} (阈值={threshold}) | {ms}ms")
                else:
                    log.info(f"🔍 [向量/本地] MISS | 无向量条目 | {ms}ms")
            else:
                log.info(f"🔍 [向量/本地] MISS | 空缓存 | {ms}ms")

        # ④ Jaccard 词重叠匹配
        if self.cfg.cache.jaccard_enabled:
            t0 = _t.time()
            result, score = await self._jaccard_search(tenant_id, clean_prompt, threshold)
            ms = int((_t.time() - t0) * 1000)
            if result is not None:
                log.info(f"🔍 [Jaccard] HIT | score={score:.4f} | {ms}ms")
                return result, score, True
            log.info(f"🔍 [Jaccard] MISS | {ms}ms")

        total = int((_t.time() - t_start) * 1000)
        log.info(f"🔍 缓存未命中 (总耗时 {total}ms)")
        return None, 0.0, False

    # ==================== 存储 ====================
    async def store(self, tenant_id: str, prompt: str, response: dict):
        """存入缓存（Redis + 内存双写）"""
        clean_prompt = clean_text(prompt)
        prompt_hash = hashlib.sha256(clean_prompt.encode()).hexdigest()

        # 生成向量
        vector = None

        # 写入 Redis
        if self.redis:
            try:
                key = f"gateway:cache:{tenant_id}:{prompt_hash}"
                data = {
                    "prompt": clean_prompt,
                    "response": json.dumps(response, ensure_ascii=False),
                }
                if vector is not None:
                    data["vector"] = vector.tobytes()

                await self.redis.hset(key, mapping=data)
                await self.redis.expire(key, self.cfg.cache.ttl_hours * 3600)
            except Exception as e:
                log.warning(f"⚠️  Redis 写入失败: {e}")

        # 生成向量
        if vector is None and self.cfg.cache.vector_enabled:
            vector = await self.embedder.embed(clean_prompt)

        # 写入内存
        entry = CacheEntry(
            prompt=clean_prompt,
            response=response,
            vector=vector,
            timestamp=time.time(),
        )

        if tenant_id not in self.local_index:
            self.local_index[tenant_id] = []

        entries = self.local_index[tenant_id]
        entries.append(entry)

        # 超过上限，淘汰最旧的
        if len(entries) > self.max_per_tenant:
            self.local_index[tenant_id] = entries[-self.max_per_tenant:]

        # 同时写入共享缓存（所有用户共享）
        if tenant_id != "_shared":
            shared_entries = self.local_index.get("_shared", [])
            shared_entries.append(entry)
            if len(shared_entries) > self.max_per_tenant:
                self.local_index["_shared"] = shared_entries[-self.max_per_tenant:]
            else:
                self.local_index["_shared"] = shared_entries

        log.info(f"💾 缓存写入: tenant={tenant_id}, hash={prompt_hash[:16]}...")

    # ==================== 去重机制 ====================
    async def _check_dedup(self, tenant_id: str, prompt: str) -> Optional[dict]:
        """
        并发去重：用 asyncio.Event 实现请求合并

        第一个请求：创建 Event，返回 None（需要调上游）
        后续并发请求：等待 Event 被 set，直接拿结果
        """
        dedup_key = f"{tenant_id}:{prompt}"

        if dedup_key in self.pending_requests:
            # 后续请求：等待第一个请求的结果
            event = self.pending_requests[dedup_key]
            try:
                await asyncio.wait_for(event.wait(), timeout=self.cfg.dedup.max_wait_seconds)
                return self._pending_results.get(dedup_key)
            except asyncio.TimeoutError:
                return None
        else:
            # 没有等待中的并发请求，直接跳过去重
            return None

    async def notify_dedup(self, tenant_id: str, prompt: str, response: dict):
        """通知所有等待中的并发请求：结果已就绪"""
        dedup_key = f"{tenant_id}:{prompt}"

        if dedup_key in self.pending_requests:
            self._pending_results[dedup_key] = response
            event = self.pending_requests.pop(dedup_key)
            event.set()

            # 清理结果缓存（给等待者留时间读取）
            await asyncio.sleep(1)
            self._pending_results.pop(dedup_key, None)

    # ==================== 精确哈希查找 ====================
    async def _exact_lookup(self, tenant_id: str, prompt_hash: str) -> Optional[dict]:
        """精确哈希匹配：SHA256 完全一致"""
        # 查 Redis
        if self.redis:
            try:
                key = f"gateway:cache:{tenant_id}:{prompt_hash}"
                response_str = await self.redis.hget(key, "response")
                if response_str:
                    return json.loads(response_str)
            except Exception:
                pass

        # 查内存
        entries = self.local_index.get(tenant_id, [])
        for entry in entries:
            entry_hash = hashlib.sha256(entry.prompt.encode()).hexdigest()
            if entry_hash == prompt_hash:
                return entry.response

        return None

    # ==================== 向量语义查找 ====================
    async def _vector_search_redis(
        self, tenant_id: str, query_vector: np.ndarray, threshold: float
    ) -> Tuple[Optional[dict], float]:
        """Redis 向量搜索：遍历 tenant 下所有 key，计算余弦相似度"""
        if not self.redis:
            return None, 0.0

        pattern = f"gateway:cache:{tenant_id}:*"
        best_match = None
        best_score = 0.0

        try:
            cursor = 0
            while True:
                cursor, keys = await self.redis.scan(cursor=cursor, match=pattern, count=100)

                # 批量读取
                pipe = self.redis.pipeline()
                for key in keys:
                    pipe.hget(key, "vector")
                    pipe.hget(key, "response")
                results = await pipe.execute()

                # 成对处理结果
                for i in range(0, len(results), 2):
                    vector_bytes = results[i]
                    response_str = results[i + 1]

                    if vector_bytes and response_str:
                        cached_vector = np.frombuffer(vector_bytes, dtype=np.float32)
                        score = cosine_similarity(query_vector, cached_vector)

                        if score > best_score and score >= threshold:
                            best_score = score
                            best_match = json.loads(response_str)

                if cursor == 0:
                    break

        except Exception as e:
            log.warning(f"⚠️  Redis 向量搜索失败: {e}")

        return best_match, best_score

    def _vector_search_local(
        self, tenant_id: str, query_vector: np.ndarray, threshold: float
    ) -> Tuple[Optional[dict], float]:
        """内存向量搜索"""
        entries = self.local_index.get(tenant_id, [])
        if not entries:
            return None, 0.0

        best_match = None
        best_score = 0.0

        for entry in entries:
            if entry.vector is None:
                continue
            score = cosine_similarity(query_vector, entry.vector)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = entry.response

        return best_match, best_score

    # ==================== Jaccard 查找 ====================
    async def _jaccard_search(
        self, tenant_id: str, target: str, threshold: float
    ) -> Tuple[Optional[dict], float]:
        """Jaccard 词重叠搜索（兜底策略）"""
        # 查 Redis
        if self.redis:
            pattern = f"gateway:cache:{tenant_id}:*"
            try:
                cursor = 0
                while True:
                    cursor, keys = await self.redis.scan(cursor=cursor, match=pattern, count=100)

                    for key in keys:
                        cached_prompt = await self.redis.hget(key, "prompt")
                        if cached_prompt:
                            score = jaccard_similarity(target, cached_prompt)
                            if score >= self.cfg.cache.jaccard_threshold:
                                response_str = await self.redis.hget(key, "response")
                                if response_str:
                                    return json.loads(response_str), score

                    if cursor == 0:
                        break
            except Exception:
                pass

        # 查内存
        entries = self.local_index.get(tenant_id, [])
        best_match = None
        best_score = 0.0

        for entry in entries:
            score = jaccard_similarity(target, entry.prompt)
            if score > best_score and score >= self.cfg.cache.jaccard_threshold:
                best_score = score
                best_match = entry.response

        return best_match, best_score

    # ==================== 统计 ====================
    async def get_stats(self) -> dict:
        """获取缓存统计信息"""
        total_entries = sum(len(entries) for entries in self.local_index.values())

        stats = {
            "local_index_entries": total_entries,
            "tenants": len(self.local_index),
            "vector_dimension": self.cfg.cache.vector_dimension,
            "similarity_threshold": self.cfg.similarity_threshold,
            "jaccard_threshold": self.cfg.cache.jaccard_threshold,
            "cache_mode": self.cfg.cache_mode,
            "ttl_hours": self.cfg.cache.ttl_hours,
        }

        # Redis 统计
        if self.redis:
            try:
                info = await self.redis.info("memory")
                stats["redis_memory_used"] = info.get("used_memory_human", "N/A")
            except Exception:
                pass

        return stats
