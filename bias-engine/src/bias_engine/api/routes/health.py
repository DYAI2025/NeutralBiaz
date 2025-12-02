"""
Health check endpoints.
"""

import time
from datetime import datetime
from typing import Dict

import structlog
from fastapi import APIRouter, status

from bias_engine.core.config import settings
from bias_engine.models.schemas import HealthResponse

logger = structlog.get_logger(__name__)
router = APIRouter()

# Track startup time
_startup_time = time.time()


async def check_dependencies() -> Dict[str, str]:
    """Check status of external dependencies."""
    dependencies = {}

    # Check Redis (if configured)
    try:
        # This would be implemented when Redis service is added
        dependencies["redis"] = "ok"
    except Exception as e:
        dependencies["redis"] = f"error: {str(e)}"

    # Check ML models
    try:
        # This would check if models are loaded
        dependencies["models"] = "ok"
    except Exception as e:
        dependencies["models"] = f"error: {str(e)}"

    # Add more dependency checks as needed
    dependencies["database"] = "not_configured"

    return dependencies


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Check the health status of the Bias Engine API and its dependencies.",
)
async def health_check() -> HealthResponse:
    """
    Perform a health check of the service.

    Returns information about:
    - Service status
    - Uptime
    - Version
    - Dependency status
    """

    logger.info("Health check requested")

    # Calculate uptime
    uptime = time.time() - _startup_time

    # Check dependencies
    dependencies = await check_dependencies()

    # Determine overall status
    status_value = "healthy"
    for dep_status in dependencies.values():
        if dep_status.startswith("error"):
            status_value = "degraded"
            break

    response = HealthResponse(
        status=status_value,
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        uptime=uptime,
        dependencies=dependencies,
    )

    logger.info(
        "Health check completed",
        status=status_value,
        uptime=uptime,
        dependencies=dependencies,
    )

    return response


@router.get(
    "/health/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness Check",
    description="Check if the service is ready to accept requests.",
)
async def readiness_check():
    """
    Check if the service is ready to handle requests.

    This is typically used by orchestrators like Kubernetes
    to determine when to start routing traffic.
    """

    logger.info("Readiness check requested")

    # Check if critical dependencies are available
    dependencies = await check_dependencies()

    # For now, we're ready if the service is running
    # In a real implementation, you might check:
    # - Models are loaded
    # - Database connections are established
    # - Required APIs are accessible

    ready = True
    failing_deps = [
        name for name, status in dependencies.items()
        if status.startswith("error") and name in ["models"]  # Critical deps only
    ]

    if failing_deps:
        ready = False

    if ready:
        return {"status": "ready", "timestamp": datetime.utcnow()}
    else:
        logger.warning("Service not ready", failing_dependencies=failing_deps)
        return {
            "status": "not_ready",
            "timestamp": datetime.utcnow(),
            "failing_dependencies": failing_deps,
        }


@router.get(
    "/health/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness Check",
    description="Check if the service is alive and responsive.",
)
async def liveness_check():
    """
    Check if the service is alive.

    This is typically used by orchestrators like Kubernetes
    to determine if the service needs to be restarted.
    """

    return {"status": "alive", "timestamp": datetime.utcnow()}