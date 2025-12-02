"""
Custom exceptions and exception handlers for the Bias Engine.
"""

import traceback
from typing import Any, Dict, Optional

import structlog
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

logger = structlog.get_logger(__name__)


class BiasEngineError(Exception):
    """Base exception for Bias Engine errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)


class ModelLoadError(BiasEngineError):
    """Raised when model loading fails."""
    pass


class AnalysisError(BiasEngineError):
    """Raised when text analysis fails."""
    pass


class ConfigurationError(BiasEngineError):
    """Raised when configuration is invalid."""
    pass


class RateLimitError(BiasEngineError):
    """Raised when rate limit is exceeded."""
    pass


class ValidationError(BiasEngineError):
    """Raised when input validation fails."""
    pass


def create_error_response(
    message: str,
    error_code: str,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    details: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    """Create standardized error response."""
    error_data = {
        "error": {
            "code": error_code,
            "message": message,
            "details": details or {},
        }
    }

    return JSONResponse(
        status_code=status_code,
        content=error_data,
    )


async def bias_engine_exception_handler(
    request: Request, exc: BiasEngineError
) -> JSONResponse:
    """Handle BiasEngineError exceptions."""
    logger.error(
        "BiasEngineError occurred",
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        path=request.url.path,
        method=request.method,
    )

    return create_error_response(
        message=exc.message,
        error_code=exc.error_code or "BIAS_ENGINE_ERROR",
        status_code=status.HTTP_400_BAD_REQUEST,
        details=exc.details,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTPException."""
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method,
    )

    return create_error_response(
        message=exc.detail,
        error_code="HTTP_ERROR",
        status_code=exc.status_code,
    )


async def validation_exception_handler(
    request: Request, exc: ValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    logger.warning(
        "Validation error occurred",
        errors=exc.errors(),
        path=request.url.path,
        method=request.method,
    )

    return create_error_response(
        message="Validation failed",
        error_code="VALIDATION_ERROR",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        details={"validation_errors": exc.errors()},
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.error(
        "Unexpected error occurred",
        error_type=type(exc).__name__,
        error_message=str(exc),
        traceback=traceback.format_exc(),
        path=request.url.path,
        method=request.method,
    )

    return create_error_response(
        message="Internal server error",
        error_code="INTERNAL_ERROR",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        details={"error_type": type(exc).__name__},
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup exception handlers for the FastAPI application."""
    app.add_exception_handler(BiasEngineError, bias_engine_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)