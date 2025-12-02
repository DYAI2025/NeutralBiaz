"""Health check endpoints for BiazNeutralize AI."""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psutil
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db_session
from ..redis import get_redis_client
from ..config import settings

router = APIRouter()


class HealthStatus(BaseModel):
    """Health status model."""
    status: str
    timestamp: datetime
    uptime: float
    version: str
    environment: str


class DetailedHealthStatus(BaseModel):
    """Detailed health status model."""
    status: str
    timestamp: datetime
    uptime: float
    version: str
    environment: str
    checks: Dict[str, Any]
    system: Dict[str, Any]
    dependencies: Dict[str, Any]


class DatabaseCheck(BaseModel):
    """Database health check model."""
    status: str
    response_time_ms: float
    connection_pool: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class RedisCheck(BaseModel):
    """Redis health check model."""
    status: str
    response_time_ms: float
    info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SystemMetrics(BaseModel):
    """System metrics model."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    load_average: List[float]
    process_count: int


# Track service start time
START_TIME = time.time()


def get_uptime() -> float:
    """Get service uptime in seconds."""
    return time.time() - START_TIME


def get_system_metrics() -> SystemMetrics:
    """Get system performance metrics."""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return SystemMetrics(
        cpu_percent=psutil.cpu_percent(interval=1),
        memory_percent=memory.percent,
        memory_used_mb=round(memory.used / 1024 / 1024, 2),
        memory_total_mb=round(memory.total / 1024 / 1024, 2),
        disk_usage_percent=disk.percent,
        disk_free_gb=round(disk.free / 1024 / 1024 / 1024, 2),
        load_average=list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0],
        process_count=len(psutil.pids())
    )


async def check_database(db: AsyncSession) -> DatabaseCheck:
    """Check database connectivity and performance."""
    start_time = time.time()
    try:
        # Simple query to test connectivity
        result = await db.execute(text("SELECT 1 as healthy"))
        row = result.fetchone()
        
        response_time_ms = round((time.time() - start_time) * 1000, 2)
        
        if row and row[0] == 1:
            return DatabaseCheck(
                status="healthy",
                response_time_ms=response_time_ms,
                connection_pool={
                    "pool_size": db.bind.pool.size(),
                    "checked_in": db.bind.pool.checkedin(),
                    "checked_out": db.bind.pool.checkedout(),
                    "overflow": db.bind.pool.overflow(),
                } if hasattr(db.bind, 'pool') else None
            )
        else:
            return DatabaseCheck(
                status="unhealthy",
                response_time_ms=response_time_ms,
                error="Invalid response from database"
            )
    except Exception as e:
        response_time_ms = round((time.time() - start_time) * 1000, 2)
        return DatabaseCheck(
            status="unhealthy",
            response_time_ms=response_time_ms,
            error=str(e)
        )


async def check_redis() -> RedisCheck:
    """Check Redis connectivity and performance."""
    start_time = time.time()
    try:
        redis_client = await get_redis_client()
        
        # Test basic connectivity
        await redis_client.ping()
        
        response_time_ms = round((time.time() - start_time) * 1000, 2)
        
        # Get Redis info
        info = await redis_client.info()
        
        return RedisCheck(
            status="healthy",
            response_time_ms=response_time_ms,
            info={
                "version": info.get("redis_version"),
                "memory_used_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0)
            }
        )
    except Exception as e:
        response_time_ms = round((time.time() - start_time) * 1000, 2)
        return RedisCheck(
            status="unhealthy",
            response_time_ms=response_time_ms,
            error=str(e)
        )


async def check_model_availability() -> Dict[str, Any]:
    """Check if required ML models are available."""
    try:
        import spacy
        
        model_checks = {}
        required_models = ["en_core_web_sm", "de_core_news_sm"]
        
        for model_name in required_models:
            try:
                nlp = spacy.load(model_name)
                model_checks[model_name] = {
                    "status": "available",
                    "version": nlp.meta.get("version", "unknown"),
                    "lang": nlp.meta.get("lang", "unknown")
                }
            except OSError:
                model_checks[model_name] = {
                    "status": "missing",
                    "error": f"Model {model_name} not found"
                }
            except Exception as e:
                model_checks[model_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return {
            "status": "available" if all(
                check["status"] == "available" 
                for check in model_checks.values()
            ) else "degraded",
            "models": model_checks
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@router.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """Basic health check endpoint."""
    return HealthStatus(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        uptime=get_uptime(),
        version=settings.VERSION or "unknown",
        environment=settings.ENVIRONMENT
    )


@router.get("/ready")
async def readiness_check(
    db: AsyncSession = Depends(get_db_session)
) -> JSONResponse:
    """Readiness check for Kubernetes."""
    try:
        # Check database connectivity
        db_check = await check_database(db)
        
        # Check Redis connectivity
        redis_check = await check_redis()
        
        if db_check.status == "healthy" and redis_check.status == "healthy":
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"status": "ready"}
            )
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "not ready",
                    "database": db_check.status,
                    "redis": redis_check.status
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not ready",
                "error": str(e)
            }
        )


@router.get("/live")
async def liveness_check() -> JSONResponse:
    """Liveness check for Kubernetes."""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "alive"}
    )


@router.get("/health/detailed", response_model=DetailedHealthStatus)
async def detailed_health_check(
    db: AsyncSession = Depends(get_db_session)
) -> DetailedHealthStatus:
    """Detailed health check with system metrics and dependencies."""
    # Run all health checks in parallel
    db_check_task = asyncio.create_task(check_database(db))
    redis_check_task = asyncio.create_task(check_redis())
    model_check_task = asyncio.create_task(check_model_availability())
    
    # Get system metrics
    system_metrics = get_system_metrics()
    
    # Wait for all checks to complete
    db_check = await db_check_task
    redis_check = await redis_check_task
    model_check = await model_check_task
    
    # Determine overall status
    checks = {
        "database": db_check.dict(),
        "redis": redis_check.dict(),
        "models": model_check
    }
    
    # Overall status logic
    if (
        db_check.status == "healthy" and 
        redis_check.status == "healthy" and 
        model_check["status"] in ["available", "degraded"]
    ):
        overall_status = "healthy"
    elif (
        db_check.status in ["healthy", "degraded"] and 
        redis_check.status in ["healthy", "degraded"]
    ):
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    return DetailedHealthStatus(
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        uptime=get_uptime(),
        version=settings.VERSION or "unknown",
        environment=settings.ENVIRONMENT,
        checks=checks,
        system=system_metrics.dict(),
        dependencies={
            "database_response_time_ms": db_check.response_time_ms,
            "redis_response_time_ms": redis_check.response_time_ms
        }
    )


@router.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        
        metrics_data = generate_latest()
        return JSONResponse(
            content=metrics_data.decode('utf-8'),
            media_type=CONTENT_TYPE_LATEST
        )
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prometheus client not available"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating metrics: {str(e)}"
        )