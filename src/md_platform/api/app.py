"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from .routes import router

logging.basicConfig(level=logging.INFO)


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
    
    app.include_router(router)
    
    @app.get("/health")
    async def health_check():
        return {"status": "ok", "version": "2.0.0"}
        
    return app

# For uvicorn: `uvicorn md_platform.api.app:app`
app = create_app()
