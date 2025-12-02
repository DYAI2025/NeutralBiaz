"""
Text analysis endpoints.
"""

import time
from typing import List

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from bias_engine.core.exceptions import AnalysisError, ValidationError
from bias_engine.models.schemas import (
    AnalysisRequest,
    AnalysisResult,
    BatchAnalysisRequest,
    BatchAnalysisResult,
    BiasDetection,
    BiasLevel,
    BiasType,
)

logger = structlog.get_logger(__name__)
router = APIRouter()


async def perform_bias_analysis(
    text: str,
    request: AnalysisRequest
) -> AnalysisResult:
    """
    Perform bias analysis on a single text.

    This is a placeholder implementation that will be replaced
    with actual ML model inference.
    """
    start_time = time.time()

    try:
        # Simulate language detection
        detected_language = request.language if request.language != "auto" else "en"

        # Mock bias detection results
        detections = []

        # Simple keyword-based mock detection for demonstration
        bias_keywords = {
            "he should": BiasType.GENDER,
            "she should": BiasType.GENDER,
            "boys are": BiasType.GENDER,
            "girls are": BiasType.GENDER,
            "old people": BiasType.AGE,
            "young people": BiasType.AGE,
        }

        text_lower = text.lower()
        for keyword, bias_type in bias_keywords.items():
            if keyword in text_lower:
                start_pos = text_lower.find(keyword)
                end_pos = start_pos + len(keyword)

                detection = BiasDetection(
                    type=bias_type,
                    level=BiasLevel.MEDIUM,
                    confidence=0.75,
                    description=f"Potential {bias_type.value} bias detected",
                    affected_text=text[start_pos:end_pos],
                    start_position=start_pos,
                    end_position=end_pos,
                    suggestions=[
                        "Consider using gender-neutral language",
                        "Replace with inclusive terms"
                    ]
                )
                detections.append(detection)

        # Calculate overall bias score
        if detections:
            overall_bias_score = sum(d.confidence for d in detections) / len(detections)
            if overall_bias_score > 0.8:
                bias_level = BiasLevel.HIGH
            elif overall_bias_score > 0.5:
                bias_level = BiasLevel.MEDIUM
            else:
                bias_level = BiasLevel.LOW
        else:
            overall_bias_score = 0.0
            bias_level = BiasLevel.LOW

        # Generate neutralized text if requested
        neutralized_text = None
        if request.include_suggestions and detections:
            neutralized_text = text  # Placeholder - would implement actual neutralization

        processing_time = time.time() - start_time

        result = AnalysisResult(
            text=text,
            language=detected_language,
            overall_bias_score=overall_bias_score,
            bias_level=bias_level,
            detections=detections,
            neutralized_text=neutralized_text,
            processing_time=processing_time,
            model_version="mock-v1.0.0"
        )

        logger.info(
            "Analysis completed",
            text_length=len(text),
            detections_count=len(detections),
            processing_time=processing_time,
            overall_bias_score=overall_bias_score
        )

        return result

    except Exception as e:
        logger.error("Analysis failed", error=str(e), text_length=len(text))
        raise AnalysisError(
            message=f"Failed to analyze text: {str(e)}",
            error_code="ANALYSIS_FAILED"
        )


@router.post(
    "/analyze",
    response_model=AnalysisResult,
    status_code=status.HTTP_200_OK,
    summary="Analyze Text for Bias",
    description="Analyze a single text for bias and provide neutralization suggestions.",
)
async def analyze_text(request: AnalysisRequest) -> AnalysisResult:
    """
    Analyze text for bias detection.

    This endpoint accepts text and analyzes it for various types of bias,
    returning detected biases with confidence scores and suggestions for
    neutralization.
    """

    logger.info(
        "Analysis request received",
        text_length=len(request.text),
        mode=request.mode,
        cultural_profile=request.cultural_profile,
        language=request.language
    )

    # Validate request
    if len(request.text.strip()) == 0:
        raise ValidationError(
            message="Text cannot be empty",
            error_code="EMPTY_TEXT"
        )

    # Perform analysis
    try:
        result = await perform_bias_analysis(request.text, request)
        return result

    except AnalysisError:
        raise
    except Exception as e:
        logger.error("Unexpected error during analysis", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during analysis"
        )


@router.post(
    "/analyze/batch",
    response_model=BatchAnalysisResult,
    status_code=status.HTTP_200_OK,
    summary="Batch Analyze Texts for Bias",
    description="Analyze multiple texts for bias in a single request.",
)
async def analyze_batch(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks
) -> BatchAnalysisResult:
    """
    Analyze multiple texts for bias in batch.

    This endpoint is optimized for processing multiple texts efficiently
    and returns comprehensive results for all inputs.
    """

    logger.info(
        "Batch analysis request received",
        texts_count=len(request.texts),
        mode=request.mode,
        cultural_profile=request.cultural_profile
    )

    start_time = time.time()
    results = []

    try:
        # Convert to single analysis requests
        for i, text in enumerate(request.texts):
            single_request = AnalysisRequest(
                text=text,
                mode=request.mode,
                cultural_profile=request.cultural_profile,
                language=request.language,
                bias_types=request.bias_types,
                confidence_threshold=request.confidence_threshold,
                include_suggestions=request.include_suggestions
            )

            try:
                result = await perform_bias_analysis(text, single_request)
                results.append(result)

            except Exception as e:
                logger.error(
                    "Failed to analyze text in batch",
                    index=i,
                    error=str(e),
                    text_preview=text[:100]
                )
                # Continue with other texts, but record error
                continue

        total_processing_time = time.time() - start_time

        # Calculate summary statistics
        total_detections = sum(len(r.detections) for r in results)
        avg_bias_score = (
            sum(r.overall_bias_score for r in results) / len(results)
            if results else 0.0
        )

        summary = {
            "total_detections": total_detections,
            "average_bias_score": avg_bias_score,
            "texts_with_bias": sum(1 for r in results if r.detections),
            "success_rate": len(results) / len(request.texts)
        }

        batch_result = BatchAnalysisResult(
            results=results,
            total_processed=len(results),
            total_processing_time=total_processing_time,
            summary=summary
        )

        logger.info(
            "Batch analysis completed",
            total_texts=len(request.texts),
            successful_analyses=len(results),
            total_processing_time=total_processing_time,
            summary=summary
        )

        return batch_result

    except Exception as e:
        logger.error("Batch analysis failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch analysis failed"
        )