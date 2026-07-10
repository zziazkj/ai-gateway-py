"""
熔断器 — 经典三态机
CLOSED（正常）→ OPEN（熔断）→ HALF-OPEN（试探）→ CLOSED（恢复）
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Callable, Optional

log = logging.getLogger("circuit_breaker")


class CircuitState(Enum):
    CLOSED = "closed"        # 正常：允许请求通过
    OPEN = "open"            # 熔断：拒绝所有请求
    HALF_OPEN = "half_open"  # 试探：允许一个请求测试恢复


class CircuitBreaker:
    """
    熔断器实现

    原理：
    1. CLOSED 状态下，每次失败计数 +1
    2. 失败次数达到阈值 → 切换到 OPEN（熔断）
    3. OPEN 状态下，经过恢复超时 → 切换到 HALF-OPEN
    4. HALF-OPEN 下，放一个请求试探：
       - 成功 → 回到 CLOSED
       - 失败 → 回到 OPEN
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self._lock = asyncio.Lock()

    async def execute(self, fn: Callable) -> Optional[any]:
        """
        执行受熔断器保护的操作

        如果熔断器是 OPEN 状态，直接返回 None（不执行操作）
        """
        async with self._lock:
            current_state = self.state

        if current_state == CircuitState.OPEN:
            # 检查是否到了恢复时间
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                log.info("🔓 熔断器: OPEN → HALF-OPEN (允许试探)")
                self.state = CircuitState.HALF_OPEN
            else:
                log.info("🚫 熔断器: OPEN，请求被拒绝")
                return None

        if current_state == CircuitState.HALF_OPEN:
            log.info("🔓 熔断器: HALF-OPEN，允许试探请求")

        try:
            result = await fn()
            await self._record_success()
            return result

        except Exception as e:
            await self._record_failure()
            raise

    async def _record_success(self):
        """记录成功：重置计数，关闭熔断"""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                log.info("✅ 熔断器: HALF-OPEN → CLOSED (恢复正常)")
            self.failure_count = 0
            self.state = CircuitState.CLOSED

    async def _record_failure(self):
        """记录失败：计数 +1，可能触发熔断"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            log.warning(
                f"⚠️  熔断器: 失败 {self.failure_count}/{self.failure_threshold}"
            )

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                log.warning("🔴 熔断器: CLOSED → OPEN (触发熔断)")

    def is_open(self) -> bool:
        """检查熔断器是否处于 OPEN 状态"""
        return self.state == CircuitState.OPEN

    def get_state(self) -> dict:
        """获取熔断器状态"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
        }
