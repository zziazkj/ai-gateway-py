# AI Gateway Python 版 🐍

基于语义缓存的大模型 API 成本优化网关，使用 Python + FastAPI 实现。

> 🎯 **核心目标**：零代码改动，将 LLM API 成本降低 40-70%

## 🏗️ 架构

```
用户应用 → AI Gateway → [四层缓存检查]
                         ↓
                    [HIT] → 直接返回（0 成本）
                         ↓
                    [MISS] → 调用上游 LLM → 缓存 → 返回
```

## ✨ 核心特性

- **四层缓存匹配**：去重 → 精确哈希 → 向量语义 → Jaccard 词重叠
- **双模向量嵌入**：优先调用智谱 Embedding API（真正的语义理解），无 API Key 时降级为本地随机投影
- **Redis + 内存双存储**：Redis 可用时持久化，不可用时自动降级
- **并发请求去重**：100 个相同并发请求 → 只调 1 次上游
- **熔断保护**：上游故障时自动切断，防止雪崩
- **成本追踪**：实时查看缓存命中率和成本节省
- **管理面板**：内置 Web UI，可视化查看缓存状态和成本报表
- **灵活配置**：支持 `.env` + `gateway.yaml` 双配置，兼容任何 OpenAI 兼容 API

## 📁 项目结构

```
ai-gateway-py/
├── main.py              # 入口 + HTTP 路由
├── config.py            # YAML + .env 配置加载
├── embedding.py         # 向量嵌入引擎（智谱 API + 本地随机投影）
├── cache.py             # 四层语义缓存（Redis + 内存）
├── proxy.py             # 上游代理 + 重试 + 指数退避
├── circuit_breaker.py   # 熔断器（三态机）
├── cost_tracker.py      # 成本追踪
├── gateway.yaml         # YAML 配置文件
├── .env                 # 环境变量配置
├── .env.example         # 环境变量示例
├── requirements.txt     # Python 依赖
├── test_api.py          # API 连通性测试脚本
├── test_local_gateway.py # Python SDK 测试示例
└── app/
    └── static/
        └── index.html   # 管理面板 UI
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

复制 `.env.example` 为 `.env`，填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env`：

```ini
UPSTREAM_API_KEY=your-api-key-here
UPSTREAM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
DEFAULT_MODEL=glm-4-flash
EMBEDDING_MODEL=embedding-3
GATEWAY_PORT=8080
```

或通过环境变量设置：

```bash
# Windows PowerShell
$env:UPSTREAM_API_KEY="你的API密钥"

# Linux/Mac
export UPSTREAM_API_KEY=你的API密钥
```

缓存、限流、熔断等高级配置见 `gateway.yaml`。

### 3. 启动

```bash
python main.py
```

Gateway 运行在 `http://localhost:8080`，管理面板在 `http://localhost:8080/ui`

### 4. 测试

```bash
# API 连通性测试
python test_api.py

# Python SDK 测试（使用 OpenAI 库）
python test_local_gateway.py

# 健康检查
curl http://localhost:8080/health

# 发送请求（第一次：缓存未命中）
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Gateway-Token: my-app" \
  -d '{
    "model": "glm-4-flash",
    "messages": [{"role": "user", "content": "什么是RAG？"}]
  }'

# 发送相同请求（第二次：缓存命中！）
# 响应头会显示: X-Gateway-Cache: HIT

# 跳过缓存（不查也不写）
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4-flash",
    "cache": false,
    "messages": [{"role": "user", "content": "今天天气怎么样？"}]
  }'
```

## 📊 API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | 主接口：带缓存的 LLM 代理（兼容 OpenAI 格式） |
| `/health` | GET | 健康检查 |
| `/stats` | GET | 缓存统计 |
| `/api/v1/cost-report` | GET | 成本报表（支持 `?tenant=` 按租户筛选） |
| `/api/v1/cache/clear` | POST | 清空缓存（支持 `?tenant=` 按租户清空） |
| `/api/v1/cache/list` | GET | 查看缓存条目（包含问答内容） |
| `/api/v1/request-logs` | GET | 请求日志（支持 `?limit=` 限制条数） |
| `/api/v1/request-logs/clear` | POST | 清空请求日志 |
| `/ui` | GET | 管理面板 Web UI |

### 请求参数

在请求 body 中可传入以下额外字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `cache` | boolean | 设为 `false` 可跳过缓存（不查也不写入） |

## 🔍 响应头说明

```http
# 缓存命中时
X-Gateway-Cache: HIT          # 缓存命中
X-Gateway-Similarity: 0.92    # 相似度 92%
X-Gateway-Time-Saved: 5ms     # 节省的时间

# 缓存未命中时
X-Gateway-Cache: MISS          # 缓存未命中
X-Gateway-Duration: 1200ms     # 总耗时
```

## 🧠 核心算法

### 向量嵌入（embedding.py）

**双模式**，优先使用智谱 Embedding API，不可用时降级：

- **API 模式**：调用智谱 `embedding-3` 模型，输出高维向量后截取前 N 维（默认 2048），L2 归一化
- **本地降级**：随机投影（Johnson-Lindenstrauss 变换），文本分词 → 投影矩阵映射 → 向量求和 → L2 归一化

两种模式均支持向量缓存（最多 10000 条），避免重复计算。

### 四层缓存匹配（cache.py）

```
① 去重检查   → 并发相同请求等待第一个结果（避免重复调用上游）
② 精确哈希   → SHA256 完全一致，直接命中（含跨租户共享缓存）
③ 向量语义   → 余弦相似度 ≥ 0.85（默认），支持 Redis 和内存两种后端
④ Jaccard    → 词集交并比 ≥ 0.75（兜底匹配）
```

### 熔断器（circuit_breaker.py）

```
CLOSED（正常）→ 失败 ≥ 5 次 → OPEN（熔断）
OPEN（熔断）→ 超时 30s → HALF-OPEN（试探）
HALF-OPEN → 成功 → CLOSED（恢复）
```

### 上游代理（proxy.py）

- 支持任何 OpenAI 兼容 API（通过 `UPSTREAM_BASE_URL` 配置）
- 指数退避重试（500ms, 1000ms, 2000ms...）
- 熔断器集成，连续失败自动切断

## ⚙️ 配置说明

### .env 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `UPSTREAM_API_KEY` | - | 上游 LLM API 密钥 |
| `UPSTREAM_BASE_URL` | `https://open.bigmodel.cn/api/paas/v4` | 上游 API 地址 |
| `DEFAULT_MODEL` | `glm-4-flash` | 默认模型 |
| `EMBEDDING_MODEL` | `embedding-3` | 嵌入模型 |
| `GATEWAY_PORT` | `8080` | 网关端口 |

### gateway.yaml 配置

详见 `gateway.yaml`，支持配置缓存策略、向量维度、相似度阈值、去重、限流、熔断器等。

## 🎓 学习价值

这个项目涵盖了以下技术点：

1. **向量相似度计算**：余弦相似度、Embedding API 集成、随机投影降级
2. **缓存设计**：多层匹配、去重、TTL 过期、跨租户共享
3. **并发控制**：asyncio、请求合并、事件通知
4. **容错设计**：熔断器、指数退避重试
5. **API 网关**：反向代理、限流、多租户
6. **可观测性**：成本追踪、缓存统计、管理面板
