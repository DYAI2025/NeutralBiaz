"""
Model information endpoints.
"""

import structlog
from fastapi import APIRouter, status

from bias_engine.core.config import settings
from bias_engine.models.schemas import BiasType, ModelInfo, ModelsResponse

logger = structlog.get_logger(__name__)
router = APIRouter()


def get_available_models() -> list[ModelInfo]:
    """
    Get list of available models.

    In a real implementation, this would query the model registry
    or check what models are available in the models directory.
    """

    # Mock model information - replace with actual model discovery
    models = [
        ModelInfo(
            name="distilbert-bias-detector",
            version="1.0.0",
            description="DistilBERT-based bias detection model optimized for speed",
            supported_languages=["en", "es", "fr"],
            supported_bias_types=[
                BiasType.GENDER,
                BiasType.RACIAL,
                BiasType.AGE,
                BiasType.POLITICAL,
            ],
            accuracy=0.87,
            is_default=True,
            loaded=True,
        ),
        ModelInfo(
            name="bert-comprehensive-bias",
            version="2.1.0",
            description="Large BERT model for comprehensive bias detection",
            supported_languages=["en", "es", "fr", "de"],
            supported_bias_types=list(BiasType),
            accuracy=0.93,
            is_default=False,
            loaded=False,
        ),
        ModelInfo(
            name="roberta-cultural-bias",
            version="1.5.0",
            description="RoBERTa model specialized for cultural bias detection",
            supported_languages=["en"],
            supported_bias_types=[
                BiasType.CULTURAL,
                BiasType.RELIGIOUS,
                BiasType.SOCIOECONOMIC,
            ],
            accuracy=0.89,
            is_default=False,
            loaded=False,
        ),
        ModelInfo(
            name="multilingual-bias-detector",
            version="1.2.0",
            description="Multilingual transformer for global bias detection",
            supported_languages=["en", "es", "fr", "de", "zh", "ja", "ar"],
            supported_bias_types=[
                BiasType.GENDER,
                BiasType.RACIAL,
                BiasType.CULTURAL,
                BiasType.RELIGIOUS,
            ],
            accuracy=0.84,
            is_default=False,
            loaded=False,
        ),
    ]

    return models


@router.get(
    "/models",
    response_model=ModelsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Available Models",
    description="Retrieve information about available bias detection models.",
)
async def get_models() -> ModelsResponse:
    """
    Get available bias detection models.

    Returns information about:
    - Available models and their capabilities
    - Model versions and accuracy scores
    - Supported languages and bias types
    - Loading status
    - Default model configuration
    """

    logger.info("Models information request received")

    models = get_available_models()

    response = ModelsResponse(
        models=models,
        default_model=settings.default_model,
        total_models=len(models),
    )

    logger.info(
        "Models information response prepared",
        total_models=len(models),
        loaded_models=sum(1 for m in models if m.loaded),
        default_model=settings.default_model,
    )

    return response


@router.get(
    "/models/{model_name}",
    response_model=ModelInfo,
    status_code=status.HTTP_200_OK,
    summary="Get Model Information",
    description="Get detailed information about a specific model.",
)
async def get_model_info(model_name: str) -> ModelInfo:
    """
    Get information about a specific model.

    Args:
        model_name: Name of the model to get information about

    Returns:
        Detailed model information including capabilities and status
    """

    logger.info("Model information request received", model_name=model_name)

    models = get_available_models()

    # Find the requested model
    for model in models:
        if model.name == model_name:
            logger.info(
                "Model information found",
                model_name=model_name,
                version=model.version,
                loaded=model.loaded,
            )
            return model

    # Model not found
    logger.warning("Model not found", model_name=model_name)
    from fastapi import HTTPException
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Model '{model_name}' not found"
    )