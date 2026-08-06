"""FastAPI application factory."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import settings
from ..demo import available_datasets
from . import demo_routes, routes
from .rate_limit import RateLimitMiddleware

logger = logging.getLogger("md_ai_analyzer")

VERSION = "3.0.0"

DESCRIPTION = """Deterministic classical MD analysis with a grounded reporting layer.

Classical modules compute every scientific number; the reporting layer reads only
precomputed statistics through three tools, and a deterministic checker verifies
every numeric claim in a narrative against the analysis bundle before a human can
approve it.
"""


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    logging.basicConfig(
        level=logging.DEBUG if settings.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    app = FastAPI(title=settings.app_name, version=VERSION, description=DESCRIPTION)

    app.add_middleware(
        RateLimitMiddleware,
        requests_per_window=settings.rate_limit_requests,
        expensive_per_window=settings.rate_limit_expensive_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        # No cookies or auth headers are used, and credentialed wildcard CORS is
        # never safe, so credentials stay off.
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(routes.router)
    app.include_router(demo_routes.router)

    @app.get("/health", tags=["ops"])
    async def health_check():
        """Liveness probe, also reporting how this instance is configured."""
        return {
            "status": "ok",
            "version": VERSION,
            "narrator": settings.llm_model if settings.llm_enabled else "template",
            "demo_datasets": [dataset.key for dataset in available_datasets()],
            "limits": {
                "max_upload_mb": settings.max_upload_bytes // (1024 * 1024),
                "max_frames": settings.max_frames,
                "rate_limit_per_minute": settings.rate_limit_requests,
            },
        }

    return app


# For uvicorn: `uvicorn md_platform.api.app:app`
app = create_app()
