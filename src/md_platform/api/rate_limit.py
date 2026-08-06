"""Rate limiting.

A fixed-window in-process counter per client IP, with a tighter budget for the
endpoints that start work (uploads, analysis submissions, report generation)
than for polling. In-process is the right scope here: the deployment target is
one container on one small host, and a shared-state limiter would mean running
Redis for a demo.
"""

import time
from collections import defaultdict
from threading import Lock
from typing import Dict, Tuple

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Endpoint path fragments that consume real CPU or spend money.
EXPENSIVE_FRAGMENTS = ("/submit", "/upload", "/report", "/demo")


class FixedWindowLimiter:
    """Per-key request counter over a fixed time window."""

    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self._counts: Dict[str, Tuple[int, float]] = defaultdict(lambda: (0, 0.0))
        self._lock = Lock()

    def check(self, key: str) -> Tuple[bool, int]:
        """Return ``(allowed, retry_after_seconds)`` and count the request."""
        now = time.monotonic()
        with self._lock:
            count, window_start = self._counts[key]
            if now - window_start >= self.window_seconds:
                self._counts[key] = (1, now)
                return True, 0
            if count >= self.limit:
                return False, int(self.window_seconds - (now - window_start)) + 1
            self._counts[key] = (count + 1, window_start)
            return True, 0


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Applies the general and expensive-endpoint budgets."""

    def __init__(
        self,
        app,
        requests_per_window: int,
        expensive_per_window: int,
        window_seconds: int,
    ):
        super().__init__(app)
        self.general = FixedWindowLimiter(requests_per_window, window_seconds)
        self.expensive = FixedWindowLimiter(expensive_per_window, window_seconds)

    async def dispatch(self, request: Request, call_next):
        key = self._client_key(request)
        path = request.url.path

        allowed, retry_after = self.general.check(key)
        if allowed and self._is_expensive(request.method, path):
            allowed, retry_after = self.expensive.check(key)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        "Rate limit exceeded for this demo deployment. "
                        f"Retry in {retry_after}s."
                    )
                },
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)

    @staticmethod
    def _is_expensive(method: str, path: str) -> bool:
        if method not in ("POST", "PUT"):
            return False
        return any(fragment in path for fragment in EXPENSIVE_FRAGMENTS)

    @staticmethod
    def _client_key(request: Request) -> str:
        # Fly.io/Render terminate TLS upstream, so the direct peer is the proxy;
        # the left-most forwarded address is the closest thing to the caller.
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
