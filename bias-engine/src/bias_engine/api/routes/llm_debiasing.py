"""
API routes for LLM debiasing functionality.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ...llm import (
    DebiasingPipeline,
    SelfBiasChecker,
    DebiasingRequest,
    DebiasingResponse,
    BatchDebiasingRequest,
    BatchDebiasingResponse,
    MarkerGenerationRequest,
    MarkerGenerationResponse,
    SelfBiasCheckRequest,
    SelfBiasCheckResponse,
    BiasSpan,
    LLMConfig,
    RateLimitConfig
)
from ...llm.config import config_manager
from ...llm.cultural_integration import CulturalContextIntegrator
from ...core.exceptions import BiasEngineError
from ...models.schemas import BiasDetection, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["LLM Debiasing"])


class DebiasingSingleRequest(BaseModel):
    """Request model for single span debiasing."""
    bias_detection: BiasDetection = Field(description="Bias detection to neutralize")
    full_text: str = Field(description="Complete text containing the bias")
    input_language: str = Field(default="de", description="Input text language")
    output_language: str = Field(default="de", description="Output language")
    sender_culture: str = Field(default="neutral", description="Sender culture code")
    receiver_culture: str = Field(default="neutral", description="Receiver culture code")
    context_topic: str = Field(default="general", description="Context or topic")
    audience: str = Field(default="general", description="Target audience")
    formality_level: str = Field(default="neutral", description="Formality level")
    llm_provider: Optional[str] = Field(default=None, description="Specific LLM provider")


class DebiasingSingleResponse(BaseModel):
    """Response model for single span debiasing."""
    success: bool = Field(description="Whether debiasing succeeded")
    result: Optional[DebiasingResponse] = Field(description="Debiasing result")
    error: Optional[str] = Field(description="Error message if failed")
    quality_issues: List[str] = Field(default_factory=list, description="Quality validation issues")


class DebiasingBatchRequestAPI(BaseModel):
    """API request model for batch debiasing."""
    bias_detections: List[BiasDetection] = Field(description="List of bias detections")
    full_text: str = Field(description="Complete document text")
    input_language: str = Field(default="de", description="Input text language")
    output_language: str = Field(default="de", description="Output language")
    sender_culture: str = Field(default="neutral", description="Sender culture code")
    receiver_culture: str = Field(default="neutral", description="Receiver culture code")
    context_topic: str = Field(default="general", description="Context or topic")
    audience: str = Field(default="general", description="Target audience")
    formality_level: str = Field(default="neutral", description="Formality level")
    llm_provider: Optional[str] = Field(default=None, description="Specific LLM provider")


class DebiasingBatchResponseAPI(BaseModel):
    """API response model for batch debiasing."""
    success: bool = Field(description="Whether batch debiasing succeeded")
    result: Optional[BatchDebiasingResponse] = Field(description="Batch debiasing result")
    error: Optional[str] = Field(description="Error message if failed")
    quality_summary: dict = Field(default_factory=dict, description="Quality validation summary")


class MarkerGenerationRequestAPI(BaseModel):
    """API request model for marker generation."""
    bias_family: str = Field(description="Bias category family")
    bias_subtype: str = Field(description="Specific bias subtype")
    bias_description: str = Field(description="Description of bias pattern")
    output_language: str = Field(default="de", description="Target language")
    domain: str = Field(default="general", description="Application domain")
    primary_cultures: List[str] = Field(default=["de"], description="Primary cultural contexts")
    old_markers: Optional[List[dict]] = Field(default=None, description="Existing markers")
    llm_provider: Optional[str] = Field(default=None, description="Specific LLM provider")


class SelfBiasCheckRequestAPI(BaseModel):
    """API request model for self-bias checking."""
    texts: List[str] = Field(description="Texts to check for self-bias")
    context: str = Field(default="", description="Context information")
    llm_provider: Optional[str] = Field(default=None, description="Specific LLM provider")


class SelfBiasCheckResponseAPI(BaseModel):
    """API response model for self-bias checking."""
    success: bool = Field(description="Whether self-bias checking succeeded")
    results: List[SelfBiasCheckResponse] = Field(description="Self-bias check results")
    statistics: dict = Field(default_factory=dict, description="Processing statistics")
    error: Optional[str] = Field(description="Error message if failed")


async def get_debiasing_pipeline(llm_provider: Optional[str] = None) -> DebiasingPipeline:
    """Dependency to get debiasing pipeline with specified provider."""
    try:
        llm_config = config_manager.get_config(llm_provider)
        rate_limit_config = config_manager.get_rate_limit_config()

        return DebiasingPipeline(llm_config, rate_limit_config)
    except Exception as e:
        logger.error(f"Failed to create debiasing pipeline: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize LLM pipeline: {str(e)}"
        )


async def get_self_bias_checker(llm_provider: Optional[str] = None) -> SelfBiasChecker:
    """Dependency to get self-bias checker with specified provider."""
    try:
        llm_config = config_manager.get_config(llm_provider)
        rate_limit_config = config_manager.get_rate_limit_config()

        pipeline = DebiasingPipeline(llm_config, rate_limit_config)
        return SelfBiasChecker(pipeline.llm_client)
    except Exception as e:
        logger.error(f"Failed to create self-bias checker: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize self-bias checker: {str(e)}"
        )


@router.post("/debias/single", response_model=DebiasingSingleResponse)
async def debias_single_span(
    request: DebiasingSingleRequest,
    pipeline: DebiasingPipeline = Depends(get_debiasing_pipeline)
):
    """
    Debias a single text span with A/B variants.

    This endpoint takes a bias detection and generates two alternative formulations:
    - Variant A: Neutral and factual
    - Variant B: Emotionally similar but bias-reduced
    """
    try:
        # Create cultural integrator
        cultural_integrator = CulturalContextIntegrator()

        # Convert bias detection to bias span with cultural context
        bias_span = cultural_integrator.create_bias_span(
            detection=request.bias_detection,
            full_text=request.full_text,
            sender_culture=request.sender_culture,
            receiver_culture=request.receiver_culture,
            context=request.context_topic
        )

        # Create debiasing request
        debias_request = DebiasingRequest(
            bias_span=bias_span,
            input_language=request.input_language,
            output_language=request.output_language,
            sender_culture=request.sender_culture,
            receiver_culture=request.receiver_culture,
            context_topic=request.context_topic,
            audience=request.audience,
            formality_level=request.formality_level
        )

        # Perform debiasing
        async with pipeline:
            result = await pipeline.debias_span(debias_request)

            # Validate response quality
            is_valid, quality_issues = await pipeline.validate_response_quality(result)

            return DebiasingSingleResponse(
                success=True,
                result=result,
                quality_issues=quality_issues
            )

    except BiasEngineError as e:
        logger.error(f"Debiasing failed: {e}")
        return DebiasingSingleResponse(
            success=False,
            error=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in single debiasing: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Debiasing failed: {str(e)}"
        )


@router.post("/debias/batch", response_model=DebiasingBatchResponseAPI)
async def debias_batch_spans(
    request: DebiasingBatchRequestAPI,
    background_tasks: BackgroundTasks,
    pipeline: DebiasingPipeline = Depends(get_debiasing_pipeline)
):
    """
    Debias multiple text spans in a batch operation.

    Processes multiple bias detections efficiently in a single LLM call,
    maintaining document context and consistency across spans.
    """
    try:
        # Create cultural integrator
        cultural_integrator = CulturalContextIntegrator()

        # Convert all bias detections to bias spans
        bias_spans = cultural_integrator.batch_create_bias_spans(
            detections=request.bias_detections,
            full_text=request.full_text,
            sender_culture=request.sender_culture,
            receiver_culture=request.receiver_culture,
            context=request.context_topic
        )

        if not bias_spans:
            return DebiasingBatchResponseAPI(
                success=False,
                error="No valid bias spans could be created from detections"
            )

        # Create batch debiasing request
        batch_request = BatchDebiasingRequest(
            spans=bias_spans,
            full_document_text=request.full_text,
            input_language=request.input_language,
            output_language=request.output_language,
            sender_culture=request.sender_culture,
            receiver_culture=request.receiver_culture,
            context_topic=request.context_topic,
            audience=request.audience,
            formality_level=request.formality_level
        )

        # Perform batch debiasing
        async with pipeline:
            result = await pipeline.debias_batch(batch_request)

            # Validate response quality for each span
            quality_issues = []
            for span_result in result.spans:
                is_valid, issues = await pipeline.validate_response_quality(span_result)
                quality_issues.extend(issues)

            quality_summary = {
                "total_spans": len(result.spans),
                "quality_issues": len(quality_issues),
                "success_rate": (len(result.spans) - len(quality_issues)) / len(result.spans) if result.spans else 0
            }

            # Log quality metrics in background
            background_tasks.add_task(
                log_batch_metrics,
                request.bias_detections,
                result,
                quality_summary
            )

            return DebiasingBatchResponseAPI(
                success=True,
                result=result,
                quality_summary=quality_summary
            )

    except BiasEngineError as e:
        logger.error(f"Batch debiasing failed: {e}")
        return DebiasingBatchResponseAPI(
            success=False,
            error=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in batch debiasing: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch debiasing failed: {str(e)}"
        )


@router.post("/markers/generate", response_model=MarkerGenerationResponse)
async def generate_bias_markers(
    request: MarkerGenerationRequestAPI,
    pipeline: DebiasingPipeline = Depends(get_debiasing_pipeline)
):
    """
    Generate new bias detection markers for a specific bias category.

    Creates structured markers with examples and counter-examples
    to improve bias detection accuracy for specific domains and cultures.
    """
    try:
        # Create marker generation request
        marker_request = MarkerGenerationRequest(
            bias_family=request.bias_family,
            bias_subtype=request.bias_subtype,
            bias_description=request.bias_description,
            output_language=request.output_language,
            domain=request.domain,
            primary_cultures=request.primary_cultures,
            old_markers=request.old_markers
        )

        # Generate markers
        async with pipeline:
            result = await pipeline.generate_markers(marker_request)

            return result

    except BiasEngineError as e:
        logger.error(f"Marker generation failed: {e}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in marker generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Marker generation failed: {str(e)}"
        )


@router.post("/self-bias/check", response_model=SelfBiasCheckResponseAPI)
async def check_self_bias(
    request: SelfBiasCheckRequestAPI,
    checker: SelfBiasChecker = Depends(get_self_bias_checker)
):
    """
    Check texts for self-bias and apply epistemic classification.

    Analyzes texts for overconfidence, bias indicators, and applies
    appropriate epistemic prefixes according to German specification.
    """
    try:
        results = []

        for text in request.texts:
            check_request = SelfBiasCheckRequest(
                text=text,
                context=request.context
            )

            result = await checker.check_bias(check_request)
            results.append(result)

        # Calculate statistics
        statistics = checker.get_classification_statistics(results)

        return SelfBiasCheckResponseAPI(
            success=True,
            results=results,
            statistics=statistics
        )

    except BiasEngineError as e:
        logger.error(f"Self-bias checking failed: {e}")
        return SelfBiasCheckResponseAPI(
            success=False,
            results=[],
            error=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in self-bias checking: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Self-bias checking failed: {str(e)}"
        )


@router.get("/providers")
async def list_available_providers():
    """List available LLM providers and their status."""
    try:
        providers = config_manager.get_available_providers()
        provider_status = {}

        for provider in providers:
            try:
                config = config_manager.get_config(provider)
                provider_status[provider.value] = {
                    "available": True,
                    "model": config.model,
                    "has_api_key": bool(config.api_key)
                }
            except Exception as e:
                provider_status[provider.value] = {
                    "available": False,
                    "error": str(e)
                }

        return {
            "providers": provider_status,
            "default_provider": config_manager.settings.default_provider.value
        }

    except Exception as e:
        logger.error(f"Failed to list providers: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve provider information"
        )


@router.get("/cultural/recommendations")
async def get_cultural_recommendations(
    sender_culture: str,
    receiver_culture: str
):
    """Get cultural communication recommendations."""
    try:
        integrator = CulturalContextIntegrator()
        recommendations = integrator.get_cultural_recommendations(
            sender_culture=sender_culture,
            receiver_culture=receiver_culture
        )

        return recommendations

    except Exception as e:
        logger.error(f"Failed to get cultural recommendations: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate cultural recommendations"
        )


async def log_batch_metrics(
    detections: List[BiasDetection],
    result: BatchDebiasingResponse,
    quality_summary: dict
):
    """Background task to log batch processing metrics."""
    try:
        metrics = {
            "input_detections": len(detections),
            "processed_spans": result.total_processed,
            "processing_time": result.processing_time,
            "quality_issues": quality_summary.get("quality_issues", 0),
            "success_rate": quality_summary.get("success_rate", 0),
            "provider": result.batch_metadata.get("llm_provider"),
            "model": result.batch_metadata.get("llm_model"),
            "tokens_used": result.batch_metadata.get("tokens_used", {})
        }

        logger.info(f"Batch debiasing metrics: {metrics}")

        # Here you would typically send metrics to your monitoring system

    except Exception as e:
        logger.error(f"Failed to log batch metrics: {e}")