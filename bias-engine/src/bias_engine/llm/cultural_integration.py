"""
Cultural context integration with existing bias detection system.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple

from .models import BiasSpan, CulturalSeverityLevel
from ..models.schemas import BiasDetection, CulturalProfile
from ..core.exceptions import BiasEngineError

logger = logging.getLogger(__name__)


class CulturalIntegrationError(BiasEngineError):
    """Base exception for cultural integration errors."""
    pass


class CulturalContextIntegrator:
    """
    Integrates cultural context with the bias detection system.

    Provides cultural severity adjustment and context-aware bias interpretation
    based on sender and receiver cultural profiles.
    """

    def __init__(self):
        """Initialize the cultural context integrator."""
        self.cultural_mappings = self._load_cultural_mappings()
        self.severity_adjustments = self._load_severity_adjustments()

    def _load_cultural_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Load cultural mappings and characteristics."""
        return {
            "de": {
                "name": "German",
                "directness": 0.8,  # High directness tolerance
                "hierarchy_sensitivity": 0.6,  # Moderate hierarchy awareness
                "collectivism": 0.4,  # Individualistic tendency
                "context_dependency": 0.3,  # Low context culture
                "politeness_formality": 0.7,  # High formality expectation
                "bias_sensitivities": {
                    "racism": 0.9,  # Very sensitive
                    "sexism": 0.8,
                    "ageism": 0.6,
                    "classism": 0.7,
                    "ableism": 0.8
                }
            },
            "us": {
                "name": "American",
                "directness": 0.7,
                "hierarchy_sensitivity": 0.4,
                "collectivism": 0.3,
                "context_dependency": 0.3,
                "politeness_formality": 0.5,
                "bias_sensitivities": {
                    "racism": 0.95,
                    "sexism": 0.85,
                    "ageism": 0.7,
                    "classism": 0.6,
                    "ableism": 0.8
                }
            },
            "jp": {
                "name": "Japanese",
                "directness": 0.2,  # Low directness, high indirectness
                "hierarchy_sensitivity": 0.9,  # Very high hierarchy awareness
                "collectivism": 0.8,  # Highly collectivistic
                "context_dependency": 0.9,  # High context culture
                "politeness_formality": 0.95,  # Extremely high formality
                "bias_sensitivities": {
                    "racism": 0.7,
                    "sexism": 0.6,  # Different cultural norms
                    "ageism": 0.3,  # Age hierarchy is cultural norm
                    "classism": 0.5,
                    "ableism": 0.7
                }
            },
            "gb": {
                "name": "British",
                "directness": 0.4,  # Indirect communication style
                "hierarchy_sensitivity": 0.6,
                "collectivism": 0.4,
                "context_dependency": 0.6,
                "politeness_formality": 0.8,
                "bias_sensitivities": {
                    "racism": 0.9,
                    "sexism": 0.85,
                    "ageism": 0.7,
                    "classism": 0.8,  # Class awareness in UK
                    "ableism": 0.8
                }
            },
            "fr": {
                "name": "French",
                "directness": 0.7,
                "hierarchy_sensitivity": 0.7,
                "collectivism": 0.5,
                "context_dependency": 0.5,
                "politeness_formality": 0.8,
                "bias_sensitivities": {
                    "racism": 0.8,
                    "sexism": 0.8,
                    "ageism": 0.6,
                    "classism": 0.7,
                    "ableism": 0.7
                }
            },
            "es": {
                "name": "Spanish",
                "directness": 0.6,
                "hierarchy_sensitivity": 0.7,
                "collectivism": 0.6,
                "context_dependency": 0.6,
                "politeness_formality": 0.7,
                "bias_sensitivities": {
                    "racism": 0.8,
                    "sexism": 0.7,
                    "ageism": 0.5,
                    "classism": 0.7,
                    "ableism": 0.7
                }
            },
            "neutral": {
                "name": "Neutral/Global",
                "directness": 0.5,
                "hierarchy_sensitivity": 0.5,
                "collectivism": 0.5,
                "context_dependency": 0.5,
                "politeness_formality": 0.6,
                "bias_sensitivities": {
                    "racism": 0.8,
                    "sexism": 0.8,
                    "ageism": 0.6,
                    "classism": 0.6,
                    "ableism": 0.7
                }
            }
        }

    def _load_severity_adjustments(self) -> Dict[str, Dict[str, float]]:
        """Load severity adjustment matrices for cultural pairs."""
        return {
            # Adjustment factors for cross-cultural communication
            "directness_mismatch": {
                "high_to_low": 1.3,  # Direct to indirect culture
                "low_to_high": 0.8,  # Indirect to direct culture
                "moderate": 1.0
            },
            "hierarchy_mismatch": {
                "high_to_low": 1.2,
                "low_to_high": 0.9,
                "moderate": 1.0
            },
            "formality_mismatch": {
                "informal_to_formal": 1.4,
                "formal_to_informal": 0.7,
                "moderate": 1.0
            }
        }

    def create_bias_span(
        self,
        detection: BiasDetection,
        full_text: str,
        sender_culture: str = "neutral",
        receiver_culture: str = "neutral",
        context: Optional[str] = None
    ) -> BiasSpan:
        """
        Create a BiasSpan from a BiasDetection with cultural context.

        Args:
            detection: Original bias detection
            full_text: Complete text containing the bias
            sender_culture: Sender culture code
            receiver_culture: Receiver culture code
            context: Additional context information

        Returns:
            BiasSpan with cultural severity adjustments
        """
        # Calculate cultural severity adjustments
        severity_level = self.calculate_cultural_severity(
            bias_type=detection.type.value,
            raw_severity=self._convert_level_to_score(detection.level),
            sender_culture=sender_culture,
            receiver_culture=receiver_culture,
            context=context or ""
        )

        # Extract the relevant sentence or paragraph
        full_sentence = self._extract_context_sentence(
            full_text,
            detection.start_position,
            detection.end_position
        )

        return BiasSpan(
            span_id=f"span_{detection.start_position}_{detection.end_position}",
            full_sentence_or_paragraph=full_sentence,
            bias_span=detection.affected_text,
            bias_family=detection.type.value,
            bias_subtype=self._determine_bias_subtype(detection),
            severity=severity_level,
            start_position=detection.start_position,
            end_position=detection.end_position
        )

    def calculate_cultural_severity(
        self,
        bias_type: str,
        raw_severity: float,
        sender_culture: str,
        receiver_culture: str,
        context: str = ""
    ) -> CulturalSeverityLevel:
        """
        Calculate culturally-adjusted severity levels.

        Args:
            bias_type: Type of bias detected
            raw_severity: Original severity score (0-10)
            sender_culture: Sender culture code
            receiver_culture: Receiver culture code
            context: Additional context for adjustment

        Returns:
            Cultural severity level with adjustments
        """
        sender_profile = self.cultural_mappings.get(sender_culture.lower(), self.cultural_mappings["neutral"])
        receiver_profile = self.cultural_mappings.get(receiver_culture.lower(), self.cultural_mappings["neutral"])

        # Base severity adjustments based on cultural bias sensitivities
        sender_sensitivity = sender_profile["bias_sensitivities"].get(bias_type, 0.7)
        receiver_sensitivity = receiver_profile["bias_sensitivities"].get(bias_type, 0.7)

        sender_severity = raw_severity * sender_sensitivity
        receiver_severity = raw_severity * receiver_sensitivity

        # Apply cross-cultural adjustment factors
        cultural_adjustment = self._calculate_cross_cultural_adjustment(
            sender_profile, receiver_profile
        )

        receiver_severity *= cultural_adjustment

        # Ensure scores stay within bounds
        sender_severity = max(0, min(10, sender_severity))
        receiver_severity = max(0, min(10, receiver_severity))

        # Generate cultural explanation
        explanation = self._generate_cultural_explanation(
            bias_type, raw_severity, sender_severity, receiver_severity,
            sender_culture, receiver_culture, cultural_adjustment
        )

        return CulturalSeverityLevel(
            sender_culture=sender_culture,
            receiver_culture=receiver_culture,
            raw_severity=raw_severity,
            sender_severity=sender_severity,
            receiver_severity=receiver_severity,
            cultural_explanation=explanation
        )

    def _calculate_cross_cultural_adjustment(
        self,
        sender_profile: Dict[str, Any],
        receiver_profile: Dict[str, Any]
    ) -> float:
        """Calculate adjustment factor for cross-cultural communication."""
        adjustment = 1.0

        # Directness mismatch adjustment
        directness_diff = abs(sender_profile["directness"] - receiver_profile["directness"])
        if directness_diff > 0.3:
            if sender_profile["directness"] > receiver_profile["directness"]:
                adjustment *= self.severity_adjustments["directness_mismatch"]["high_to_low"]
            else:
                adjustment *= self.severity_adjustments["directness_mismatch"]["low_to_high"]

        # Hierarchy sensitivity adjustment
        hierarchy_diff = abs(sender_profile["hierarchy_sensitivity"] - receiver_profile["hierarchy_sensitivity"])
        if hierarchy_diff > 0.3:
            if sender_profile["hierarchy_sensitivity"] < receiver_profile["hierarchy_sensitivity"]:
                adjustment *= self.severity_adjustments["hierarchy_mismatch"]["low_to_high"]

        # Formality adjustment
        formality_diff = abs(sender_profile["politeness_formality"] - receiver_profile["politeness_formality"])
        if formality_diff > 0.3:
            if sender_profile["politeness_formality"] < receiver_profile["politeness_formality"]:
                adjustment *= self.severity_adjustments["formality_mismatch"]["informal_to_formal"]

        return adjustment

    def _generate_cultural_explanation(
        self,
        bias_type: str,
        raw_severity: float,
        sender_severity: float,
        receiver_severity: float,
        sender_culture: str,
        receiver_culture: str,
        adjustment: float
    ) -> str:
        """Generate explanation for cultural severity adjustments."""
        explanation_parts = []

        if sender_culture != receiver_culture:
            explanation_parts.append(f"Cross-cultural communication detected ({sender_culture} → {receiver_culture}).")

        if abs(sender_severity - receiver_severity) > 1.0:
            if receiver_severity > sender_severity:
                explanation_parts.append(
                    f"Higher sensitivity in receiver culture increases perceived severity "
                    f"({sender_severity:.1f} → {receiver_severity:.1f})."
                )
            else:
                explanation_parts.append(
                    f"Lower sensitivity in receiver culture decreases perceived severity "
                    f"({sender_severity:.1f} → {receiver_severity:.1f})."
                )

        if adjustment > 1.1:
            explanation_parts.append("Cultural communication style differences amplify potential impact.")
        elif adjustment < 0.9:
            explanation_parts.append("Cultural compatibility reduces potential negative impact.")

        sender_name = self.cultural_mappings.get(sender_culture, {}).get("name", sender_culture)
        receiver_name = self.cultural_mappings.get(receiver_culture, {}).get("name", receiver_culture)

        if not explanation_parts:
            return f"Standard {bias_type} assessment for {sender_name} to {receiver_name} communication."

        return " ".join(explanation_parts)

    def _extract_context_sentence(self, full_text: str, start_pos: int, end_pos: int) -> str:
        """Extract the sentence or paragraph containing the bias span."""
        # Find sentence boundaries around the bias span
        sentence_start = max(0, full_text.rfind('.', 0, start_pos) + 1)
        sentence_end = full_text.find('.', end_pos)
        if sentence_end == -1:
            sentence_end = len(full_text)
        else:
            sentence_end += 1

        # Extract and clean the sentence
        sentence = full_text[sentence_start:sentence_end].strip()

        # If sentence is too short, expand to paragraph
        if len(sentence) < 50:
            para_start = max(0, full_text.rfind('\n\n', 0, start_pos) + 2)
            para_end = full_text.find('\n\n', end_pos)
            if para_end == -1:
                para_end = len(full_text)

            sentence = full_text[para_start:para_end].strip()

        return sentence

    def _determine_bias_subtype(self, detection: BiasDetection) -> str:
        """Determine the specific bias subtype based on detection characteristics."""
        # Map confidence and description to subtypes
        if detection.confidence > 0.9:
            return "explicit_bias"
        elif detection.confidence > 0.7:
            return "implicit_bias"
        else:
            return "potential_bias"

        # Could be enhanced with more sophisticated subtype detection
        # based on the detection description and affected text

    def _convert_level_to_score(self, level) -> float:
        """Convert bias level enum to numeric score."""
        from ..models.schemas import BiasLevel

        level_mapping = {
            BiasLevel.LOW: 3.0,
            BiasLevel.MEDIUM: 6.0,
            BiasLevel.HIGH: 8.5,
            BiasLevel.CRITICAL: 10.0
        }

        return level_mapping.get(level, 5.0)

    def batch_create_bias_spans(
        self,
        detections: List[BiasDetection],
        full_text: str,
        sender_culture: str = "neutral",
        receiver_culture: str = "neutral",
        context: Optional[str] = None
    ) -> List[BiasSpan]:
        """
        Create multiple BiasSpans from a list of detections.

        Args:
            detections: List of bias detections
            full_text: Complete text
            sender_culture: Sender culture code
            receiver_culture: Receiver culture code
            context: Additional context

        Returns:
            List of BiasSpans with cultural adjustments
        """
        spans = []

        for detection in detections:
            try:
                span = self.create_bias_span(
                    detection=detection,
                    full_text=full_text,
                    sender_culture=sender_culture,
                    receiver_culture=receiver_culture,
                    context=context
                )
                spans.append(span)
            except Exception as e:
                logger.error(f"Failed to create bias span for detection: {e}")
                continue

        return spans

    def get_cultural_recommendations(
        self,
        sender_culture: str,
        receiver_culture: str
    ) -> Dict[str, Any]:
        """
        Get cultural communication recommendations.

        Args:
            sender_culture: Sender culture code
            receiver_culture: Receiver culture code

        Returns:
            Dictionary with cultural recommendations
        """
        sender_profile = self.cultural_mappings.get(sender_culture.lower(), self.cultural_mappings["neutral"])
        receiver_profile = self.cultural_mappings.get(receiver_culture.lower(), self.cultural_mappings["neutral"])

        recommendations = {
            "sender_culture": sender_culture,
            "receiver_culture": receiver_culture,
            "recommendations": []
        }

        # Directness recommendations
        if sender_profile["directness"] > receiver_profile["directness"] + 0.2:
            recommendations["recommendations"].append(
                "Consider using more indirect communication styles to avoid appearing harsh or aggressive."
            )

        # Hierarchy recommendations
        if sender_profile["hierarchy_sensitivity"] < receiver_profile["hierarchy_sensitivity"] - 0.2:
            recommendations["recommendations"].append(
                "Be mindful of hierarchical relationships and show appropriate respect for seniority/authority."
            )

        # Formality recommendations
        if sender_profile["politeness_formality"] < receiver_profile["politeness_formality"] - 0.2:
            recommendations["recommendations"].append(
                "Use more formal language and titles to show respect for cultural communication norms."
            )

        # Context recommendations
        if sender_profile["context_dependency"] < receiver_profile["context_dependency"] - 0.2:
            recommendations["recommendations"].append(
                "Provide more contextual information and avoid assumptions about shared understanding."
            )

        return recommendations