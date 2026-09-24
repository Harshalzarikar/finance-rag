"""Fixed-window rate limiting backed by Redis.

Implemented directly rather than through a decorator-based framework so the
limiter reads validated settings at request time and can be overridden in tests.
It fails open: if Redis is unreachable, requests are allowed and a warning is
logged, because an unavailable cache should not take the API down.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import lru_cache

import redis

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 60


@dataclass
class RateLimitResult:
    """Outcome of a single rate-limit check."""

    allowed: bool
    remaining: int
    reset_seconds: int


class RateLimiter:
    """Counts requests per identity in fixed one-minute windows."""

    def __init__(self, client: redis.Redis | None, limit: int, window_seconds: int = _WINDOW_SECONDS) -> None:
        self.client = client
        self.limit = limit
        self.window_seconds = window_seconds

    def check(self, identity: str) -> RateLimitResult:
        """Consume one unit of ``identity``'s budget for the current window."""
        now = int(time.time())
        window_start = now - (now % self.window_seconds)
        reset_seconds = window_start + self.window_seconds - now

        if self.client is None:
            return RateLimitResult(allowed=True, remaining=self.limit, reset_seconds=reset_seconds)

        bucket = f"ratelimit:{identity}:{window_start}"
        try:
            pipeline = self.client.pipeline()
            pipeline.incr(bucket)
            pipeline.expire(bucket, self.window_seconds)
            count = int(pipeline.execute()[0])
        except Exception as exc:  # noqa: BLE001 - limiter outage must not block traffic
            logger.warning("Rate limiter unavailable (%s) — allowing request.", exc)
            return RateLimitResult(allowed=True, remaining=self.limit, reset_seconds=reset_seconds)

        return RateLimitResult(
            allowed=count <= self.limit,
            remaining=max(0, self.limit - count),
            reset_seconds=reset_seconds,
        )


@lru_cache(maxsize=1)
def get_rate_limiter() -> RateLimiter:
    """Build the process-wide limiter, tolerating an unreachable Redis."""
    settings = get_settings()
    client: redis.Redis | None = None
    try:
        client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        client.ping()
        logger.info("Rate-limiter Redis is reachable.")
    except Exception as exc:  # noqa: BLE001 - misconfigured URL degrades to unlimited
        logger.warning("Rate-limiter Redis is unreachable (%s) — limits disabled until reconnected.", exc)
        client = None

    return RateLimiter(client, limit=settings.rate_limit_per_minute)
