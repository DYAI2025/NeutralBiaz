"""
Self-bias check system with epistemic classification.
"""

import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .client import BaseLLMClient
from .prompts import PromptManager
from .models import (
    SelfBiasCheckRequest,
    SelfBiasCheckResponse,
    EpistemicClassification
)
from ..core.exceptions import BiasEngineError

logger = logging.getLogger(__name__)


class SelfBiasError(BiasEngineError):
    """Base exception for self-bias checking errors."""
    pass


class SelfBiasChecker:
    """
    Self-bias checking system that applies epistemic classification and
    overconfidence detection to LLM outputs.

    Implements the German specification for epistemic prefixes:
    - "Faktisch korrekt sage ich, dass..." for objective facts
    - "Logisch scheint mir, dass..." for rational arguments
    - "Rein subjektiv, aus meinem Denken ergibt sich..." for opinions
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_manager: Optional[PromptManager] = None
    ):
        """Initialize the self-bias checker.

        Args:
            llm_client: LLM client for processing
            prompt_manager: Custom prompt manager (uses default if None)
        """
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager or PromptManager()

    async def check_bias(self, request: SelfBiasCheckRequest) -> SelfBiasCheckResponse:
        """
        Check text for self-bias and apply epistemic classification.

        Args:
            request: Self-bias check request

        Returns:
            Self-bias check response with corrected text
        """
        try:
            # Prepare template variables
            variables = {
                "text": request.text,
                "context": request.context
            }

            # Create messages
            messages = [
                self.prompt_manager.create_message("self_bias_check", variables)
            ]

            # Make LLM request
            response = await self.llm_client.generate(messages)

            # Parse JSON response
            try:
                result_data = json.loads(response.content)
            except json.JSONDecodeError as e:
                # Fallback to manual processing if JSON parsing fails
                return self._fallback_processing(request)

            # Validate and create response
            return SelfBiasCheckResponse(
                original_text=result_data.get("original_text", request.text),
                epistemic_classification=EpistemicClassification(
                    result_data.get("epistemic_classification", "subjektiv")
                ),
                overconfidence_detected=result_data.get("overconfidence_detected", False),
                bias_indicators=result_data.get("bias_indicators", []),
                corrected_text=result_data.get("corrected_text", request.text),
                confidence_score=result_data.get("confidence_score", 0.7),
                explanation=result_data.get("explanation", "Automated classification applied")
            )

        except Exception as e:
            logger.error(f"Self-bias checking failed: {e}")
            return self._fallback_processing(request)

    def _fallback_processing(self, request: SelfBiasCheckRequest) -> SelfBiasCheckResponse:
        """
        Fallback processing when LLM processing fails.
        Uses rule-based approach for basic epistemic classification.
        """
        text = request.text.strip()

        # Detect epistemic classification using patterns
        classification, bias_indicators = self._classify_epistemically(text)

        # Check for overconfidence patterns
        overconfidence = self._detect_overconfidence(text)

        # Apply epistemic prefix
        corrected_text = self._apply_epistemic_prefix(text, classification)

        return SelfBiasCheckResponse(
            original_text=text,
            epistemic_classification=classification,
            overconfidence_detected=overconfidence,
            bias_indicators=bias_indicators,
            corrected_text=corrected_text,
            confidence_score=0.6,  # Lower confidence for fallback
            explanation="Fallback rule-based classification applied"
        )

    def _classify_epistemically(self, text: str) -> Tuple[EpistemicClassification, List[str]]:
        """
        Classify text epistemically using pattern matching.

        Returns:
            Tuple of (classification, bias_indicators)
        """
        text_lower = text.lower()
        bias_indicators = []

        # Factual indicators
        factual_patterns = [
            r'\b(laut|nach|gemäß|entsprechend)\b.*\b(studie|forschung|daten|statistik)',
            r'\b(messbar|nachweisbar|belegbar|dokumentiert)\b',
            r'\b\d+\s*(prozent|%|euro|dollar|meter|kilogramm)\b',
            r'\bist\s+(ein|eine|der|die|das)\s+fakt\b',
        ]

        logical_patterns = [
            r'\b(daraus folgt|daher|deshalb|folglich|somit)\b',
            r'\b(wenn.*dann|falls.*dann)\b',
            r'\b(logisch|rational|vernünftig)\b.*\b(dass|zu)\b',
            r'\b(schlussfolgerung|ableitung|konsequenz)\b',
        ]

        subjective_patterns = [
            r'\b(ich finde|ich denke|ich glaube|ich meine)\b',
            r'\b(meiner meinung|aus meiner sicht)\b',
            r'\b(vermutlich|wahrscheinlich|möglicherweise)\b',
            r'\b(gefühl|eindruck|empfindung)\b',
        ]

        # Check for overconfidence indicators
        overconfident_patterns = [
            r'\b(definitiv|absolut|zweifellos|unbestreitbar|eindeutig)\b',
            r'\b(alle|niemand|nie|immer|jeder|niemals)\b',
            r'\b(es ist klar|offensichtlich|selbstverständlich)\b',
        ]

        # Count matches for each category
        factual_score = sum(1 for pattern in factual_patterns if re.search(pattern, text_lower))
        logical_score = sum(1 for pattern in logical_patterns if re.search(pattern, text_lower))
        subjective_score = sum(1 for pattern in subjective_patterns if re.search(pattern, text_lower))

        # Check for bias indicators
        for pattern in overconfident_patterns:
            if re.search(pattern, text_lower):
                bias_indicators.append(f"Overconfident language: {pattern}")

        # Determine classification
        if factual_score > logical_score and factual_score > subjective_score:
            return EpistemicClassification.FAKTISCH, bias_indicators
        elif logical_score > subjective_score:
            return EpistemicClassification.LOGISCH, bias_indicators
        else:
            return EpistemicClassification.SUBJEKTIV, bias_indicators

    def _detect_overconfidence(self, text: str) -> bool:
        """
        Detect overconfidence patterns in text.

        Returns:
            True if overconfidence detected
        """
        overconfident_patterns = [
            r'\b(definitiv|absolut|zweifellos|unbestreitbar|eindeutig|sicher)\b',
            r'\b(alle|niemand|nie|immer|jeder|niemals|garantiert)\b',
            r'\b(es ist klar|offensichtlich|selbstverständlich|natürlich)\b',
            r'\b(ohne zweifel|mit sicherheit|ganz bestimmt)\b',
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in overconfident_patterns)

    def _apply_epistemic_prefix(self, text: str, classification: EpistemicClassification) -> str:
        """
        Apply appropriate epistemic prefix to text.

        Args:
            text: Original text
            classification: Epistemic classification

        Returns:
            Text with epistemic prefix
        """
        # Check if text already has an epistemic prefix
        existing_prefixes = [
            r'^faktisch korrekt sage ich,?\s*dass',
            r'^logisch scheint mir,?\s*dass',
            r'^rein subjektiv,?\s*aus meinem denken ergibt sich',
        ]

        text_lower = text.lower()
        if any(re.search(pattern, text_lower) for pattern in existing_prefixes):
            # Text already has prefix, return as-is
            return text

        # Apply appropriate prefix
        if classification == EpistemicClassification.FAKTISCH:
            if text.lower().startswith('dass '):
                prefix = "Faktisch korrekt sage ich, "
            else:
                prefix = "Faktisch korrekt sage ich, dass "
        elif classification == EpistemicClassification.LOGISCH:
            if text.lower().startswith('dass '):
                prefix = "Logisch scheint mir, "
            else:
                prefix = "Logisch scheint mir, dass "
        else:  # SUBJEKTIV
            prefix = "Rein subjektiv, aus meinem Denken ergibt sich, dass "

        # Ensure proper sentence structure
        if text.lower().startswith('dass '):
            corrected = prefix + text[5:]  # Remove redundant "dass"
        else:
            corrected = prefix + text.lower()

        # Capitalize first letter after prefix
        words = corrected.split()
        if len(words) > 6:  # Account for prefix words
            words[6] = words[6].capitalize()
            corrected = ' '.join(words)

        return corrected

    def validate_epistemic_prefix(self, text: str) -> Tuple[bool, Optional[EpistemicClassification]]:
        """
        Validate if text has correct epistemic prefix.

        Args:
            text: Text to validate

        Returns:
            Tuple of (has_valid_prefix, detected_classification)
        """
        text_lower = text.lower().strip()

        # Define prefix patterns
        patterns = {
            EpistemicClassification.FAKTISCH: r'^faktisch korrekt sage ich,?\s*dass',
            EpistemicClassification.LOGISCH: r'^logisch scheint mir,?\s*dass',
            EpistemicClassification.SUBJEKTIV: r'^rein subjektiv,?\s*aus meinem denken ergibt sich,?\s*dass'
        }

        # Check for valid prefixes
        for classification, pattern in patterns.items():
            if re.search(pattern, text_lower):
                return True, classification

        return False, None

    def batch_check_bias(self, texts: List[str], context: str = "") -> List[SelfBiasCheckResponse]:
        """
        Process multiple texts for self-bias checking.

        Args:
            texts: List of texts to check
            context: Shared context for all texts

        Returns:
            List of self-bias check responses
        """
        results = []

        for text in texts:
            request = SelfBiasCheckRequest(text=text, context=context)
            try:
                # For batch processing, use fallback method for efficiency
                result = self._fallback_processing(request)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process text in batch: {e}")
                # Create minimal error response
                results.append(SelfBiasCheckResponse(
                    original_text=text,
                    epistemic_classification=EpistemicClassification.SUBJEKTIV,
                    overconfidence_detected=False,
                    bias_indicators=["Processing failed"],
                    corrected_text=f"Rein subjektiv, aus meinem Denken ergibt sich, dass {text.lower()}",
                    confidence_score=0.1,
                    explanation="Error during processing, fallback applied"
                ))

        return results

    def get_classification_statistics(self, responses: List[SelfBiasCheckResponse]) -> Dict[str, any]:
        """
        Get statistics from a batch of self-bias check responses.

        Args:
            responses: List of responses to analyze

        Returns:
            Dictionary with statistics
        """
        total = len(responses)
        if total == 0:
            return {}

        classifications = [r.epistemic_classification for r in responses]
        overconfidence_count = sum(1 for r in responses if r.overconfidence_detected)
        avg_confidence = sum(r.confidence_score for r in responses) / total

        stats = {
            "total_processed": total,
            "classifications": {
                "faktisch": classifications.count(EpistemicClassification.FAKTISCH),
                "logisch": classifications.count(EpistemicClassification.LOGISCH),
                "subjektiv": classifications.count(EpistemicClassification.SUBJEKTIV),
            },
            "overconfidence_detected": overconfidence_count,
            "overconfidence_rate": overconfidence_count / total,
            "average_confidence": avg_confidence,
            "bias_indicators": {
                indicator: sum(1 for r in responses if indicator in r.bias_indicators)
                for indicator in set().union(*[r.bias_indicators for r in responses])
            }
        }

        return stats