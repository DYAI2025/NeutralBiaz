"""
Debiasing pipeline with text neutralization and A/B variants.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from .client import LLMClient, BaseLLMClient
from .prompts import PromptManager
from .models import (
    DebiasingRequest,
    DebiasingResponse,
    BatchDebiasingRequest,
    BatchDebiasingResponse,
    MarkerGenerationRequest,
    MarkerGenerationResponse,
    BiasSpan,
    LLMConfig,
    RateLimitConfig
)
from ..core.exceptions import BiasEngineError

logger = logging.getLogger(__name__)


class DebiasingError(BiasEngineError):
    """Base exception for debiasing pipeline errors."""
    pass


class ValidationError(DebiasingError):
    """Raised when LLM response validation fails."""
    pass


class DebiasingPipeline:
    """
    Main debiasing pipeline that orchestrates LLM calls for bias neutralization.

    Handles:
    - Single span debiasing with A/B variants
    - Batch processing of multiple spans
    - Marker generation for bias detection
    - Response validation and quality control
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        rate_limit_config: Optional[RateLimitConfig] = None,
        prompt_manager: Optional[PromptManager] = None
    ):
        """Initialize the debiasing pipeline.

        Args:
            llm_config: LLM client configuration
            rate_limit_config: Rate limiting configuration
            prompt_manager: Custom prompt manager (uses default if None)
        """
        self.llm_config = llm_config
        self.rate_limit_config = rate_limit_config
        self.prompt_manager = prompt_manager or PromptManager()

        # Create LLM client
        self.llm_client = LLMClient.create(llm_config, rate_limit_config)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if hasattr(self.llm_client, '__aexit__'):
            await self.llm_client.__aexit__(exc_type, exc_val, exc_tb)

    async def debias_span(self, request: DebiasingRequest) -> DebiasingResponse:
        """
        Debias a single text span with A/B variants.

        Args:
            request: Debiasing request for a single span

        Returns:
            Debiasing response with neutralized variants

        Raises:
            DebiasingError: If debiasing fails
            ValidationError: If response validation fails
        """
        start_time = datetime.utcnow()

        try:
            # Prepare template variables
            variables = {
                "span_id": request.bias_span.span_id,
                "input_language": request.input_language,
                "output_language": request.output_language,
                "sender_culture": request.sender_culture,
                "receiver_culture": request.receiver_culture,
                "context_topic": request.context_topic,
                "audience": request.audience,
                "formality_level": request.formality_level,
                "full_sentence_or_paragraph": request.bias_span.full_sentence_or_paragraph,
                "bias_span": request.bias_span.bias_span,
                "bias_family": request.bias_span.bias_family,
                "bias_subtype": request.bias_span.bias_subtype,
                "severity_raw": request.bias_span.severity.raw_severity,
                "severity_sender": request.bias_span.severity.sender_severity,
                "severity_receiver": request.bias_span.severity.receiver_severity,
                "cultural_explanation": request.bias_span.severity.cultural_explanation,
            }

            # Create messages
            messages = [
                self.prompt_manager.create_message("debiaser_system", {"output_language": request.output_language}),
                self.prompt_manager.create_message("debias_span", variables)
            ]

            # Make LLM request
            response = await self.llm_client.generate(messages)

            # Parse JSON response
            try:
                result_data = json.loads(response.content)
            except json.JSONDecodeError as e:
                raise ValidationError(f"Invalid JSON response: {e}")

            # Validate required fields
            required_fields = [
                "span_id", "language", "bias_family", "bias_subtype",
                "analysis_explanation", "can_preserve_core_intent",
                "variant_A_rewrite", "variant_B_rewrite", "safety_notes"
            ]

            missing_fields = [field for field in required_fields if field not in result_data]
            if missing_fields:
                raise ValidationError(f"Missing required fields: {missing_fields}")

            # Create response object
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            return DebiasingResponse(
                span_id=result_data["span_id"],
                language=result_data["language"],
                bias_family=result_data["bias_family"],
                bias_subtype=result_data["bias_subtype"],
                analysis_explanation=result_data["analysis_explanation"],
                can_preserve_core_intent=result_data["can_preserve_core_intent"],
                variant_a_rewrite=result_data["variant_A_rewrite"],
                variant_b_rewrite=result_data["variant_B_rewrite"],
                safety_notes=result_data["safety_notes"],
                confidence_score=self._calculate_confidence_score(result_data, response),
                processing_metadata={
                    "llm_provider": response.provider.value,
                    "llm_model": response.model,
                    "tokens_used": response.tokens_used,
                    "llm_response_time": response.response_time,
                    "processing_time": processing_time,
                    "timestamp": start_time.isoformat()
                }
            )

        except Exception as e:
            logger.error(f"Span debiasing failed for {request.bias_span.span_id}: {e}")
            if isinstance(e, (DebiasingError, ValidationError)):
                raise
            raise DebiasingError(f"Debiasing pipeline failed: {e}")

    async def debias_batch(self, request: BatchDebiasingRequest) -> BatchDebiasingResponse:
        """
        Debias multiple spans in a batch operation.

        Args:
            request: Batch debiasing request

        Returns:
            Batch debiasing response
        """
        start_time = datetime.utcnow()

        try:
            # Prepare spans data for template
            spans_data = []
            for span in request.spans:
                spans_data.append({
                    "span_id": span.span_id,
                    "full_sentence_or_paragraph": span.full_sentence_or_paragraph,
                    "bias_span": span.bias_span,
                    "bias_family": span.bias_family,
                    "bias_subtype": span.bias_subtype,
                    "severity_raw": span.severity.raw_severity,
                    "severity_sender": span.severity.sender_severity,
                    "severity_receiver": span.severity.receiver_severity,
                    "cultural_explanation": span.severity.cultural_explanation
                })

            # Prepare template variables
            variables = {
                "input_language": request.input_language,
                "output_language": request.output_language,
                "sender_culture": request.sender_culture,
                "receiver_culture": request.receiver_culture,
                "context_topic": request.context_topic,
                "audience": request.audience,
                "formality_level": request.formality_level,
                "full_document_text": request.full_document_text,
                "spans_json": json.dumps(spans_data, ensure_ascii=False, indent=2)
            }

            # Create messages
            messages = [
                self.prompt_manager.create_message("debiaser_system", {"output_language": request.output_language}),
                self.prompt_manager.create_message("debias_batch", variables)
            ]

            # Make LLM request
            response = await self.llm_client.generate(messages)

            # Parse JSON response
            try:
                result_data = json.loads(response.content)
            except json.JSONDecodeError as e:
                raise ValidationError(f"Invalid JSON response: {e}")

            # Validate response structure
            if "spans" not in result_data or not isinstance(result_data["spans"], list):
                raise ValidationError("Response must contain 'spans' array")

            # Process individual span responses
            span_responses = []
            for span_data in result_data["spans"]:
                try:
                    span_response = DebiasingResponse(
                        span_id=span_data["span_id"],
                        language=result_data.get("language", request.output_language),
                        bias_family=span_data["bias_family"],
                        bias_subtype=span_data["bias_subtype"],
                        analysis_explanation=span_data["analysis_explanation"],
                        can_preserve_core_intent=span_data["can_preserve_core_intent"],
                        variant_a_rewrite=span_data["variant_A_rewrite"],
                        variant_b_rewrite=span_data["variant_B_rewrite"],
                        safety_notes=span_data["safety_notes"],
                        confidence_score=self._calculate_confidence_score(span_data, response),
                        processing_metadata={
                            "batch_processed": True,
                            "llm_provider": response.provider.value,
                            "llm_model": response.model
                        }
                    )
                    span_responses.append(span_response)
                except Exception as e:
                    logger.warning(f"Failed to process span {span_data.get('span_id', 'unknown')}: {e}")
                    continue

            processing_time = (datetime.utcnow() - start_time).total_seconds()

            return BatchDebiasingResponse(
                language=result_data.get("language", request.output_language),
                spans=span_responses,
                total_processed=len(span_responses),
                processing_time=processing_time,
                batch_metadata={
                    "llm_provider": response.provider.value,
                    "llm_model": response.model,
                    "tokens_used": response.tokens_used,
                    "llm_response_time": response.response_time,
                    "requested_spans": len(request.spans),
                    "successful_spans": len(span_responses),
                    "timestamp": start_time.isoformat()
                }
            )

        except Exception as e:
            logger.error(f"Batch debiasing failed: {e}")
            if isinstance(e, (DebiasingError, ValidationError)):
                raise
            raise DebiasingError(f"Batch debiasing pipeline failed: {e}")

    async def generate_markers(self, request: MarkerGenerationRequest) -> MarkerGenerationResponse:
        """
        Generate new bias detection markers.

        Args:
            request: Marker generation request

        Returns:
            Generated markers response
        """
        start_time = datetime.utcnow()

        try:
            # Prepare template variables
            variables = {
                "output_language": request.output_language,
                "domain": request.domain,
                "primary_cultures": json.dumps(request.primary_cultures),
                "bias_family": request.bias_family,
                "bias_subtype": request.bias_subtype,
                "bias_description": request.bias_description,
                "old_markers_json": json.dumps(request.old_markers) if request.old_markers else "null"
            }

            # Create messages
            messages = [
                self.prompt_manager.create_message("debiaser_system", {"output_language": request.output_language}),
                self.prompt_manager.create_message("marker_generator", variables)
            ]

            # Make LLM request
            response = await self.llm_client.generate(messages)

            # Parse JSON response
            try:
                result_data = json.loads(response.content)
            except json.JSONDecodeError as e:
                raise ValidationError(f"Invalid JSON response: {e}")

            # Validate response structure
            required_fields = ["bias_family", "bias_subtype", "language", "markers"]
            missing_fields = [field for field in required_fields if field not in result_data]
            if missing_fields:
                raise ValidationError(f"Missing required fields: {missing_fields}")

            processing_time = (datetime.utcnow() - start_time).total_seconds()

            return MarkerGenerationResponse(
                bias_family=result_data["bias_family"],
                bias_subtype=result_data["bias_subtype"],
                language=result_data["language"],
                markers=result_data["markers"],
                generation_metadata={
                    "llm_provider": response.provider.value,
                    "llm_model": response.model,
                    "tokens_used": response.tokens_used,
                    "llm_response_time": response.response_time,
                    "processing_time": processing_time,
                    "timestamp": start_time.isoformat(),
                    "markers_generated": len(result_data["markers"])
                }
            )

        except Exception as e:
            logger.error(f"Marker generation failed: {e}")
            if isinstance(e, (DebiasingError, ValidationError)):
                raise
            raise DebiasingError(f"Marker generation pipeline failed: {e}")

    def _calculate_confidence_score(self, result_data: Dict[str, Any], llm_response: Any) -> float:
        """
        Calculate confidence score for debiasing results.

        Args:
            result_data: Parsed LLM response data
            llm_response: Raw LLM response

        Returns:
            Confidence score between 0 and 1
        """
        base_score = 0.7  # Base confidence

        # Adjust based on various factors
        adjustments = 0.0

        # Check if both variants are provided and different
        if (result_data.get("variant_A_rewrite") and
            result_data.get("variant_B_rewrite") and
            result_data["variant_A_rewrite"] != result_data["variant_B_rewrite"]):
            adjustments += 0.1

        # Check if analysis explanation is comprehensive
        explanation = result_data.get("analysis_explanation", "")
        if len(explanation) > 50:  # Reasonable explanation length
            adjustments += 0.1

        # Check if safety notes are provided when intent cannot be preserved
        if not result_data.get("can_preserve_core_intent", True):
            safety_notes = result_data.get("safety_notes", "")
            if len(safety_notes) > 20:
                adjustments += 0.05

        # Adjust based on token usage (proxy for model engagement)
        tokens_used = getattr(llm_response, 'tokens_used', {})
        completion_tokens = tokens_used.get('completion_tokens', 0)
        if completion_tokens > 200:  # Substantial response
            adjustments += 0.05

        return min(1.0, base_score + adjustments)

    async def validate_response_quality(self, response: DebiasingResponse) -> Tuple[bool, List[str]]:
        """
        Validate the quality of a debiasing response.

        Args:
            response: Debiasing response to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check if variants are actually different
        if response.variant_a_rewrite.strip() == response.variant_b_rewrite.strip():
            issues.append("Variant A and B are identical")

        # Check for obvious bias markers in the rewrites
        bias_indicators = [
            "alle [Gruppe] sind", "typisch [Gruppe]", "wie alle",
            "natürlich [Gruppe]", "bekanntlich [Gruppe]"
        ]

        for variant in [response.variant_a_rewrite, response.variant_b_rewrite]:
            for indicator in bias_indicators:
                if any(pattern in variant.lower() for pattern in indicator.split()):
                    issues.append(f"Potential bias indicator detected in variant: {indicator}")

        # Check minimum explanation length
        if len(response.analysis_explanation.strip()) < 30:
            issues.append("Analysis explanation too short")

        # Check for safety notes when intent cannot be preserved
        if not response.can_preserve_core_intent and len(response.safety_notes.strip()) < 20:
            issues.append("Insufficient safety notes for non-preservable intent")

        return len(issues) == 0, issues