"""
Main FastAPI application entry point.
"""

import logging
import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from bias_engine.api.routes import health, analyze, config, models, llm_debiasing
from bias_engine.core.config import settings
from bias_engine.core.exceptions import setup_exception_handlers
from bias_engine.core.logging_config import setup_logging


# Setup structured logging
setup_logging()
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Bias Engine API", version=settings.app_version)

    # Initialize services here if needed
    # await initialize_ml_models()
    # await setup_redis_connection()

    yield

    # Shutdown
    logger.info("Shutting down Bias Engine API")
    # Cleanup resources here if needed


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title=settings.app_name,
        description="AI-powered bias detection and neutralization engine",
        version=settings.app_version,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan,
    )

    # Security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware for request logging and timing
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()

        # Log incoming request
        logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            user_agent=request.headers.get("user-agent", ""),
        )

        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Log response
        logger.info(
            "Request completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            process_time=f"{process_time:.4f}s",
        )

        # Add processing time to response headers
        response.headers["X-Process-Time"] = str(process_time)

        return response

    # Include routers
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(analyze.router, prefix="/api/v1", tags=["analysis"])
    app.include_router(config.router, prefix="/api/v1", tags=["configuration"])
    app.include_router(models.router, prefix="/api/v1", tags=["models"])
    app.include_router(llm_debiasing.router, prefix="/api/v1", tags=["llm-debiasing"])

    # Setup exception handlers
    setup_exception_handlers(app)

    return app


# Create the FastAPI application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "bias_engine.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_config=None,  # Use our custom logging
    )