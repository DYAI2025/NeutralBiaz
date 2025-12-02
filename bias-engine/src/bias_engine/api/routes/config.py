"""
Configuration endpoints.
"""

import structlog
from fastapi import APIRouter, status

from bias_engine.core.config import settings
from bias_engine.models.schemas import (
    AnalysisMode,
    BiasType,
    ConfigResponse,
    CulturalProfile,
)

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get(
    "/config",
    response_model=ConfigResponse,
    status_code=status.HTTP_200_OK,
    summary="Get System Configuration",
    description="Retrieve current system configuration and available options.",
)
async def get_config() -> ConfigResponse:
    """
    Get system configuration.

    Returns information about:
    - Supported languages
    - Available bias types
    - Cultural profiles
    - Analysis modes
    - Default settings
    """

    logger.info("Configuration request received")

    # Get all available bias types
    bias_types = list(BiasType)

    # Get all cultural profiles
    cultural_profiles = list(CulturalProfile)

    # Get all analysis modes
    analysis_modes = list(AnalysisMode)

    # Build default settings
    default_settings = {
        "bias_threshold": settings.bias_threshold,
        "confidence_threshold": settings.confidence_threshold,
        "default_model": settings.default_model,
        "default_cultural_profile": settings.default_cultural_profile,
        "cultural_profiles_enabled": settings.cultural_profiles_enabled,
        "max_request_size": settings.max_request_size,
        "request_timeout": settings.request_timeout,
    }

    config = ConfigResponse(
        app_name=settings.app_name,
        app_version=settings.app_version,
        environment=settings.environment,
        supported_languages=settings.supported_languages,
        supported_bias_types=bias_types,
        cultural_profiles=cultural_profiles,
        analysis_modes=analysis_modes,
        default_settings=default_settings,
    )

    logger.info(
        "Configuration response prepared",
        supported_languages_count=len(settings.supported_languages),
        bias_types_count=len(bias_types),
        cultural_profiles_count=len(cultural_profiles),
    )

    return config