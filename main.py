"""
AI-Gateway Python 版 — 语义缓存大模型 API 网关
核心功能：四层缓存匹配 + Redis 持久化 + 熔断器 + 成本追踪
"""

import os
import time
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import httpx
import numpy as np
import yaml
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse

from cache import SemanticCache
from proxy import UpstreamProvider
from circuit_breaker import CircuitBreaker
from cost_tracker import CostTracker
from config import load_config

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ai-gateway")

# ========== 全局状态 ==========
cfg = load_config()
cache: Optional[SemanticCache] = None
upstream: Optional[UpstreamProvider] = None
cost_tracker = CostTracker()

# 请求日志存储（内存，最多保留100条）
request_logs = []
MAX_LOGS = 100


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化，关闭时清理"""
    global cache, upstream

    # 初始化缓存
    cache = SemanticCache(cfg)
    log.info(f"✅ 缓存初始化完成: 模式={cfg.cache_mode}")

    # 初始化上游代理
    upstream = UpstreamProvider(cfg)
    key_preview = cfg.upstream_api_key[:10] + "..." if cfg.upstream_api_key else "未设置"
    key_len = len(cfg.upstream_api_key)
    log.info(f"✅ 上游代理初始化完成: provider={cfg.upstream_provider}, key_len={key_len}, key={key_preview}")

    yield

    # 关闭清理
    if cache:
        await cache.close()
    log.info("🛑 AI Gateway 已关闭")


app = FastAPI(title="AI Gateway", version="1.0.0", lifespan=lifespan)

# ========== 静态文件（管理面板） ==========
static_dir = Path(__file__).parent / "app" / "static"
if static_dir.exists():
    @app.get("/ui", response_class=HTMLResponse)
    async def ui():
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return HTMLResponse("<h1>UI 文件未找到</h1>")


# ========== 核心：聊天补全接口 ==========
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    主接口：接收 LLM 请求，先查缓存，未命中则调上游
    """
    start_time = time.time()

    # 1. 提取租户标识
    tenant_id = request.headers.get("X-Gateway-Token", "default")
    # 直接使用 .env 中的 API Key，不从请求头提取（避免格式问题）
    user_api_key = ""

    # 2. 读取请求体
    body = await request.body()
    body_json = json.loads(body)

    # 如果客户端没传 model，用配置的默认模型
    if "model" not in body_json or not body_json["model"]:
        body_json["model"] = cfg.upstream.default_model

    # 支持 cache: false 跳过缓存
    skip_cache = body_json.pop("cache", True) is False

    # 3. 提取 prompt 用于缓存匹配
    messages = body_json.get("messages", [])
    user_prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_prompt = msg.get("content", "")
            break

    if not user_prompt:
        user_prompt = json.dumps(body_json, ensure_ascii=False)

    # 4. 缓存查找（四层匹配）
    if skip_cache:
        log.info(f"⏭️  跳过缓存 (cache=false): prompt='{user_prompt[:30]}'")
        cache_hit = False
        similarity = 0.0
    else:
        threshold = cfg.similarity_threshold
        log.info(f"🔍 搜索缓存: prompt='{user_prompt[:30]}', threshold={threshold}")
        cached_response, similarity, cache_hit = await cache.search(
            tenant_id, user_prompt, threshold
        )
        log.info(f"🔍 缓存结果: hit={cache_hit}, similarity={similarity:.4f}")

    if cache_hit and similarity >= threshold:
        # ✅ 缓存命中：直接返回
        tokens_saved = len(json.dumps(cached_response)) // 4
        cost_tracker.record_hit(tenant_id, similarity, tokens_saved)

        elapsed_ms = int((time.time() - start_time) * 1000)
        log.info(f"🔍 缓存命中: tenant={tenant_id}, 相似度={similarity:.4f}")

        # 记录请求日志
        add_request_log(user_prompt, "HIT", similarity, elapsed_ms, tenant_id)

        return JSONResponse(
            content=cached_response,
            headers={
                "X-Gateway-Cache": "HIT",
                "X-Gateway-Similarity": f"{similarity:.4f}",
                "X-Gateway-Time-Saved": f"{elapsed_ms}ms",
            },
        )

    # ❌ 缓存未命中：调用上游 LLM
    log.info(f"🔄 缓存未命中: tenant={tenant_id}, 调用上游...")
    cost_tracker.record_miss(tenant_id)

    response_data = await upstream.call(body_json, user_api_key)

    # 5. 存入缓存（跳过缓存时不写入）
    if not skip_cache:
        await cache.store(tenant_id, user_prompt, response_data)

    # 6. 唤醒去重等待的并发请求
    await cache.notify_dedup(tenant_id, user_prompt, response_data)

    elapsed_ms = int((time.time() - start_time) * 1000)

    # 记录请求日志
    add_request_log(user_prompt, "MISS", 0.0, elapsed_ms, tenant_id)

    return JSONResponse(
        content=response_data,
        headers={
            "X-Gateway-Cache": "MISS",
            "X-Gateway-Duration": f"{elapsed_ms}ms",
        },
    )


# ========== 请求日志管理 ==========
def add_request_log(prompt: str, status: str, similarity: float, duration_ms: int, tenant: str = "default"):
    """添加请求日志"""
    import datetime
    log_entry = {
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "prompt": prompt[:100],
        "status": status,
        "similarity": f"{similarity:.4f}" if similarity > 0 else "-",
        "duration": f"{duration_ms}ms",
        "tenant": tenant,
        "timestamp": time.time()
    }
    request_logs.insert(0, log_entry)
    # 保持日志数量在限制内
    if len(request_logs) > MAX_LOGS:
        request_logs.pop()


@app.get("/api/v1/request-logs")
async def get_request_logs(limit: int = 50):
    """获取请求日志"""
    return {"logs": request_logs[:limit], "total": len(request_logs)}


@app.post("/api/v1/request-logs/clear")
async def clear_request_logs():
    """清空请求日志"""
    request_logs.clear()
    return {"message": "Logs cleared"}


# ========== 健康检查 ==========
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


# ========== 缓存统计 ==========
@app.get("/stats")
async def stats():
    cache_stats = await cache.get_stats() if cache else {}
    return {"uptime": time.time(), "cache": cache_stats}


# ========== 成本报表 ==========
@app.get("/api/v1/cost-report")
async def cost_report(tenant: str = ""):
    return cost_tracker.get_report(tenant)


# ========== 缓存管理 ==========
@app.post("/api/v1/cache/clear")
async def cache_clear(tenant: str = ""):
    """清空缓存"""
    if cache:
        if tenant:
            cache.local_index.pop(tenant, None)
        else:
            cache.local_index.clear()
    return {"message": "Cache cleared", "tenant": tenant or "all"}


@app.get("/api/v1/cache/list")
async def cache_list(tenant: str = ""):
    """查看缓存条目"""
    if not cache:
        return {"entries": []}

    results = []
    tenants = [tenant] if tenant else list(cache.local_index.keys())
    for t in tenants:
        if t == "_shared":
            continue  # 跳过共享缓存，避免重复显示
        entries = cache.local_index.get(t, [])
        for e in entries:
            # 提取回答内容
            answer = ""
            if isinstance(e.response, dict):
                choices = e.response.get("choices", [])
                if choices:
                    answer = choices[0].get("message", {}).get("content", "")
            
            results.append({
                "tenant": t,
                "prompt": e.prompt[:100],
                "answer": answer[:200] if answer else "",
                "timestamp": e.timestamp,
                "has_vector": e.vector is not None,
            })
    return {"total": len(results), "entries": results}


# ========== Prometheus 指标 ==========
@app.get("/metrics")
async def metrics():
    """简易 Prometheus 指标"""
    stats_data = cost_tracker.get_report("")
    lines = [
        "# HELP ai_gateway_requests_total Total requests",
        "# TYPE ai_gateway_requests_total counter",
        f'ai_gateway_requests_total{{tenant="all"}} {stats_data["total_requests"]}',
        "",
        "# HELP ai_gateway_cache_hits_total Cache hits",
        "# TYPE ai_gateway_cache_hits_total counter",
        f'ai_gateway_cache_hits_total{{tenant="all"}} {stats_data["cache_hits"]}',
        "",
        "# HELP ai_gateway_cache_misses_total Cache misses",
        "# TYPE ai_gateway_cache_misses_total counter",
        f'ai_gateway_cache_misses_total{{tenant="all"}} {stats_data["cache_misses"]}',
        "",
        "# HELP ai_gateway_hit_rate Cache hit rate",
        "# TYPE ai_gateway_hit_rate gauge",
        f'ai_gateway_hit_rate {stats_data["hit_rate_percent"]:.2f}',
    ]
    return Response(content="\n".join(lines), media_type="text/plain")


# ========== 启动入口 ==========
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", cfg.port))
    log.info(f"🚀 AI Gateway 启动在 http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
