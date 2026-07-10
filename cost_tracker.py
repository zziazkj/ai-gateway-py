"""
成本追踪器：记录每次缓存命中/未命中的成本节省
"""

import time
import threading
from typing import List, Optional


class CostEntry:
    """单条成本记录"""
    __slots__ = ("timestamp", "tenant_id", "cache_status", "similarity", "tokens_saved", "cost_saved")

    def __init__(
        self,
        tenant_id: str,
        cache_status: str,
        similarity: float = 0.0,
        tokens_saved: int = 0,
        cost_saved: float = 0.0,
    ):
        self.timestamp = time.time()
        self.tenant_id = tenant_id
        self.cache_status = cache_status
        self.similarity = similarity
        self.tokens_saved = tokens_saved
        self.cost_saved = cost_saved


class CostTracker:
    """
    成本追踪器

    功能：
    - 记录每次请求的缓存状态和节省成本
    - 生成成本报告（按租户筛选）
    - 预估月度节省
    """

    # 每 1K token 的成本（美元）
    COST_PER_1K_TOKENS = 0.03

    def __init__(self):
        self.entries: List[CostEntry] = []
        self._lock = threading.Lock()

    def record_hit(
        self, tenant_id: str, similarity: float, tokens_saved: int
    ):
        """记录缓存命中"""
        cost_saved = tokens_saved / 1000.0 * self.COST_PER_1K_TOKENS

        with self._lock:
            self.entries.append(CostEntry(
                tenant_id=tenant_id,
                cache_status="HIT",
                similarity=similarity,
                tokens_saved=tokens_saved,
                cost_saved=cost_saved,
            ))

    def record_miss(self, tenant_id: str):
        """记录缓存未命中"""
        with self._lock:
            self.entries.append(CostEntry(
                tenant_id=tenant_id,
                cache_status="MISS",
            ))

    def get_report(self, tenant_id: str = "") -> dict:
        """生成成本报告"""
        with self._lock:
            total_hits = 0
            total_misses = 0
            total_tokens_saved = 0
            total_cost_saved = 0.0

            for entry in self.entries:
                if tenant_id and entry.tenant_id != tenant_id:
                    continue

                if entry.cache_status == "HIT":
                    total_hits += 1
                    total_tokens_saved += entry.tokens_saved
                    total_cost_saved += entry.cost_saved
                else:
                    total_misses += 1

        total_requests = total_hits + total_misses
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0.0

        return {
            "tenant_id": tenant_id or "all",
            "total_requests": total_requests,
            "cache_hits": total_hits,
            "cache_misses": total_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "total_tokens_saved": total_tokens_saved,
            "total_cost_saved_usd": round(total_cost_saved, 6),
            "estimated_monthly_savings_usd": round(total_cost_saved * 30, 4),
        }
