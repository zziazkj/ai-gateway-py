"""
向量嵌入引擎
优先使用智谱 Embedding API（真正的语义向量）
降级为本地随机投影（词频匹配）
"""

import hashlib
import math
import os
import re
from typing import List

import httpx
import numpy as np


class VectorEmbeddingEngine:
    """
    向量嵌入引擎

    - 优先调用智谱 Embedding API（embedding-3），真正理解语义
    - 无 API Key 时降级为本地随机投影（词频匹配）
    """

    def __init__(self, dimension: int = 128, vocab_size: int = 20000):
        self.dimension = dimension
        self.vocab_size = vocab_size
        self.vocab: dict[str, int] = {}
        self.use_api = False
        self._client = None
        self._vector_cache: dict[str, np.ndarray] = {}  # 文本→向量缓存

        # 尝试加载智谱 API Key
        api_key = os.getenv("UPSTREAM_API_KEY", "").strip()
        if api_key:
            self.use_api = True
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(10))
            self._api_key = api_key
            base_url = os.getenv("UPSTREAM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4").strip()
            self._api_url = f"{base_url}/embeddings"
            self._model = os.getenv("EMBEDDING_MODEL", "embedding-3").strip()
        else:
            # 降级：本地随机投影
            rng = np.random.RandomState(42)
            scale = 1.0 / math.sqrt(dimension)
            total = vocab_size * dimension
            u1 = rng.random(total)
            u2 = rng.random(total)
            z = np.sqrt(-2.0 * np.log(u1 + 1e-10)) * np.cos(2.0 * np.pi * u2)
            self.projection = (z * scale).reshape(vocab_size, dimension).astype(np.float32)

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        if not tokens:
            return []
        return sorted(set(tokens))

    def _get_token_id(self, token: str) -> int:
        if token in self.vocab:
            return self.vocab[token]
        token_id = len(self.vocab)
        if token_id >= self.vocab_size:
            token_id = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.vocab_size
        self.vocab[token] = token_id
        return token_id

    async def _embed_api(self, text: str) -> np.ndarray:
        """调用智谱 Embedding API"""
        try:
            resp = await self._client.post(
                self._api_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                json={"model": self._model, "input": text},
            )
            if resp.status_code == 200:
                data = resp.json()
                vec = np.array(data["data"][0]["embedding"], dtype=np.float32)
                # 截取前 dimension 维（保留最高信息量的部分）
                if len(vec) > self.dimension:
                    vec = vec[: self.dimension]
                elif len(vec) < self.dimension:
                    vec = np.pad(vec, (0, self.dimension - len(vec)))
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
                return vec
        except Exception:
            pass
        return self._embed_local(text)

    def _embed_local(self, text: str) -> np.ndarray:
        """本地随机投影（降级方案）"""
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.dimension, dtype=np.float32)
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in tokens:
            token_id = self._get_token_id(token)
            vector += self.projection[token_id % self.vocab_size]
        magnitude = np.linalg.norm(vector)
        if magnitude > 1e-10:
            vector /= magnitude
        return vector

    async def embed(self, text: str) -> np.ndarray:
        """文本 → 向量（优先缓存 → API → 本地）"""
        # 先查本地缓存
        if text in self._vector_cache:
            return self._vector_cache[text]

        # 调 API 或本地
        if self.use_api and self._client:
            vec = await self._embed_api(text)
        else:
            vec = self._embed_local(text)

        # 存入缓存（最多 10000 条）
        if len(self._vector_cache) < 10000:
            self._vector_cache[text] = vec

        return vec

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return [await self.embed(t) for t in texts]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度"""
    if a.shape != b.shape or a.shape[0] == 0:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Jaccard 相似度"""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def clean_text(text: str) -> str:
    """文本清洗"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    words.sort()
    return " ".join(words)
