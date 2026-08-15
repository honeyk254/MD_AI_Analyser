"""FastAPI application factory."""

import os
import time
from collections import defaultdict, deque

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse
import logging

from .routes import router, demo_router

logging.basicConfig(level=logging.INFO)

MAX_REQUEST_BODY_BYTES = int(os.getenv("MAX_REQUEST_BODY_BYTES", str(64 * 1024)))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
_request_buckets = defaultdict(deque)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="MD AI Platform",
        version="2.0.0",
        description="Phase 1: Structural Biology Analysis Infrastructure",
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_guard(request: Request, call_next):
        if request.method == "POST" and request.url.path.startswith("/api/"):
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > MAX_REQUEST_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body too large."},
                )

            forwarded_for = request.headers.get("x-forwarded-for", "")
            client_host = forwarded_for.split(",")[0].strip() if forwarded_for else ""
            if not client_host:
                client_host = request.client.host if request.client else "unknown"
            now = time.time()
            bucket = _request_buckets[client_host]
            while bucket and now - bucket[0] > 60:
                bucket.popleft()
            if len(bucket) >= RATE_LIMIT_PER_MINUTE:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded."},
                )
            bucket.append(now)

        return await call_next(request)

    app.include_router(router)
    app.include_router(demo_router)
    
    @app.get("/health")
    async def health_check():
        return {"status": "ok", "version": "2.0.0"}
        
    return app

# For uvicorn: `uvicorn md_platform.api.app:app`
app = create_app()
