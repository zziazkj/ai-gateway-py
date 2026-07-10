"""
配置管理：YAML 加载 + 默认值
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
import yaml

log = logging.getLogger("config")

# 加载 .env 文件
load_dotenv()


@dataclass
class GatewayConfig:
    """网关配置"""
    port: int = 8080
    log_level: str = "info"


@dataclass
class CacheConfig:
    """缓存配置"""
    redis_url: str = "redis://localhost:6379"
    ttl_hours: int = 168  # 7 天
    # 向量匹配
    vector_enabled: bool = True
    vector_dimension: int = 128
    similarity_threshold: float = 0.85
    # Jaccard 词重叠匹配
    jaccard_enabled: bool = True
    jaccard_threshold: float = 0.75
    # 精确哈希
    exact_hash_enabled: bool = True


@dataclass
class DedupConfig:
    """去重配置"""
    enabled: bool = True
    max_wait_seconds: int = 30


@dataclass
class RateLimitConfig:
    """限流配置"""
    enabled: bool = False
    max_requests: int = 60
    window_seconds: int = 60


@dataclass
class UpstreamConfig:
    """上游 LLM 配置"""
    provider: str = "zhipu"
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    default_model: str = "glm-4-flash"
    timeout_seconds: int = 30
    max_retries: int = 2


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 30


@dataclass
class AppConfig:
    """应用总配置"""
    port: int = 8080
    upstream_provider: str = "zhipu"
    upstream_api_key: str = ""
    similarity_threshold: float = 0.85

    # 缓存模式：redis / memory
    cache_mode: str = "memory"

    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    upstream: UpstreamConfig = field(default_factory=UpstreamConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)


def load_config(path: str = "gateway.yaml") -> AppConfig:
    """加载配置文件，缺失则使用默认值"""
    cfg = AppConfig()

    # 从 .env 读取配置
    cfg.upstream_api_key = os.getenv("UPSTREAM_API_KEY", "").strip()
    cfg.upstream.base_url = os.getenv("UPSTREAM_BASE_URL", cfg.upstream.base_url).strip()
    cfg.upstream.default_model = os.getenv("DEFAULT_MODEL", "glm-4-flash").strip()
    cfg.port = int(os.getenv("GATEWAY_PORT", str(cfg.port)))

    if not cfg.upstream_api_key:
        log.warning("⚠️  UPSTREAM_API_KEY 未设置，上游调用将失败")

    # 尝试读取 YAML 配置
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # 映射 YAML 到 dataclass
            if "gateway" in data:
                cfg.port = data["gateway"].get("port", cfg.port)

            if "cache" in data:
                c = data["cache"]
                cfg.cache.redis_url = c.get("redis_url", cfg.cache.redis_url)
                cfg.cache.ttl_hours = c.get("ttl_hours", cfg.cache.ttl_hours)
                if "vector" in c:
                    v = c["vector"]
                    cfg.cache.vector_enabled = v.get("enabled", cfg.cache.vector_enabled)
                    cfg.cache.vector_dimension = v.get("dimension", cfg.cache.vector_dimension)
                    cfg.similarity_threshold = v.get("similarity_threshold", cfg.similarity_threshold)
                if "jaccard" in c:
                    j = c["jaccard"]
                    cfg.cache.jaccard_enabled = j.get("enabled", cfg.cache.jaccard_enabled)
                    cfg.cache.jaccard_threshold = j.get("threshold", cfg.cache.jaccard_threshold)

            if "upstream" in data:
                u = data["upstream"]
                cfg.upstream.provider = u.get("provider", cfg.upstream.provider)
                cfg.upstream.base_url = u.get("base_url", cfg.upstream.base_url)

            if "deduplication" in data:
                d = data["deduplication"]
                cfg.dedup.enabled = d.get("enabled", cfg.dedup.enabled)

            if "rate_limiter" in data:
                r = data["rate_limiter"]
                cfg.rate_limit.enabled = r.get("enabled", cfg.rate_limit.enabled)
                cfg.rate_limit.max_requests = r.get("max_requests", cfg.rate_limit.max_requests)

            # 检测 Redis 是否可用
            try:
                import redis.asyncio as aioredis
                r = aioredis.from_url(cfg.cache.redis_url, socket_connect_timeout=2)
                import asyncio
                asyncio.get_event_loop().run_until_complete(r.ping())
                cfg.cache_mode = "redis"
                log.info(f"✅ Redis 连接成功: {cfg.cache.redis_url}")
            except Exception:
                cfg.cache_mode = "memory"
                log.info("ℹ️  Redis 不可用，使用内存缓存")

            log.info(f"✅ 配置加载完成: {path}")

        except Exception as e:
            log.warning(f"⚠️  配置加载失败，使用默认值: {e}")
    else:
        log.info("ℹ️  未找到配置文件，使用默认配置")

    return cfg



