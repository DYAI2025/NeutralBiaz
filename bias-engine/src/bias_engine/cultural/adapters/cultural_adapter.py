"""
Cultural Severity Adjustment Engine

Applies cultural context to bias detection results,
modifying severity scores based on cross-cultural communication dynamics.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from ..models.hofstede_model import HofstedeModel, CulturalDimensions, CulturalProfile
from ..data.profile_manager import CulturalProfileManager


@dataclass
class CulturalContext:
    """Represents the cultural context for a communication scenario."""
    sender_culture: CulturalProfile
    receiver_culture: CulturalProfile
    cultural_distance: float
    risk_level: str
    high_risk_dimensions: List[Dict]
    mitigation_strategies: List[str]


@dataclass
class BiasAdjustment:
    """Represents cultural adjustments applied to bias results."""
    original_severity: float
    adjusted_severity: float
    cultural_modifier: float
    explanation: str
    cultural_factors: List[str]


class CulturalAdapter:
    """
    Core engine for applying cultural adjustments to bias detection results.

    Uses Hofstede's cultural dimensions to modify bias severity scores
    based on cross-cultural communication context.
    """

    def __init__(self, profile_manager: Optional[CulturalProfileManager] = None):
        """
        Initialize the cultural adapter.

        Args:
            profile_manager: Optional custom profile manager
        """
        self.logger = logging.getLogger(__name__)
        self.profile_manager = profile_manager or CulturalProfileManager()
        self.hofstede_model = HofstedeModel()

        # Cultural modifier thresholds
        self.MODIFIER_THRESHOLDS = {
            "minimal": {"min": 0.95, "max": 1.05},
            "moderate": {"min": 0.90, "max": 1.15},
            "significant": {"min": 0.80, "max": 1.30},
            "major": {"min": 0.70, "max": 1.50}
        }

        # Bias type cultural sensitivities
        self.BIAS_CULTURAL_SENSITIVITY = {
            "gender": {"PDI": 0.3, "IDV": 0.2, "MAS": 0.8, "UAI": 0.1, "LTO": 0.1, "IVR": 0.2},
            "racial": {"PDI": 0.4, "IDV": 0.3, "MAS": 0.2, "UAI": 0.2, "LTO": 0.2, "IVR": 0.3},
            "age": {"PDI": 0.6, "IDV": 0.2, "MAS": 0.1, "UAI": 0.3, "LTO": 0.7, "IVR": 0.2},
            "religious": {"PDI": 0.2, "IDV": 0.4, "MAS": 0.1, "UAI": 0.5, "LTO": 0.3, "IVR": 0.8},
            "socioeconomic": {"PDI": 0.8, "IDV": 0.5, "MAS": 0.4, "UAI": 0.2, "LTO": 0.2, "IVR": 0.3},
            "political": {"PDI": 0.5, "IDV": 0.6, "MAS": 0.2, "UAI": 0.4, "LTO": 0.3, "IVR": 0.4},
            "linguistic": {"PDI": 0.3, "IDV": 0.3, "MAS": 0.1, "UAI": 0.6, "LTO": 0.2, "IVR": 0.4}
        }

    def apply_cultural_modifiers(
        self,
        bias_results: Dict[str, Any],
        sender_culture: str,
        receiver_culture: str
    ) -> Dict[str, Any]:
        """
        Apply cultural modifiers to bias detection results.

        Args:
            bias_results: Original bias detection results
            sender_culture: Sender's culture code
            receiver_culture: Receiver's culture code

        Returns:
            Culturally-adjusted bias results
        """
        try:
            # Get cultural profiles
            sender_profile = self.profile_manager.get_profile(sender_culture)
            receiver_profile = self.profile_manager.get_profile(receiver_culture)

            # Analyze cultural context
            cultural_context = self._analyze_cultural_context(sender_profile, receiver_profile)

            # Apply adjustments to each detected bias
            adjusted_results = bias_results.copy()
            adjusted_results["cultural_context"] = self._serialize_cultural_context(cultural_context)
            adjusted_results["cultural_adjustments"] = {}

            if "biases_detected" in bias_results:
                for bias_type, bias_data in bias_results["biases_detected"].items():
                    adjustment = self._calculate_bias_adjustment(
                        bias_type,
                        bias_data,
                        cultural_context
                    )

                    # Update severity score
                    if isinstance(bias_data, dict) and "severity" in bias_data:
                        adjusted_results["biases_detected"][bias_type]["original_severity"] = bias_data["severity"]
                        adjusted_results["biases_detected"][bias_type]["severity"] = adjustment.adjusted_severity
                        adjusted_results["biases_detected"][bias_type]["cultural_adjustment"] = adjustment.cultural_modifier

                    # Store adjustment details
                    adjusted_results["cultural_adjustments"][bias_type] = {
                        "original_severity": adjustment.original_severity,
                        "adjusted_severity": adjustment.adjusted_severity,
                        "cultural_modifier": adjustment.cultural_modifier,
                        "explanation": adjustment.explanation,
                        "cultural_factors": adjustment.cultural_factors
                    }

            # Update overall scores
            self._update_overall_scores(adjusted_results, cultural_context)

            return adjusted_results

        except Exception as e:
            self.logger.error(f"Error applying cultural modifiers: {e}")
            # Return original results with error note
            error_results = bias_results.copy()
            error_results["cultural_error"] = str(e)
            return error_results

    def _analyze_cultural_context(
        self,
        sender_profile: CulturalProfile,
        receiver_profile: CulturalProfile
    ) -> CulturalContext:
        """Analyze the cultural context between sender and receiver."""

        # Calculate cultural distance and risk assessment
        risk_assessment = self.hofstede_model.assess_communication_risk(
            sender_profile.dimensions,
            receiver_profile.dimensions
        )

        return CulturalContext(
            sender_culture=sender_profile,
            receiver_culture=receiver_profile,
            cultural_distance=risk_assessment["overall_distance"],
            risk_level=risk_assessment["risk_level"],
            high_risk_dimensions=risk_assessment["high_risk_dimensions"],
            mitigation_strategies=risk_assessment["mitigation_strategies"]
        )

    def _calculate_bias_adjustment(
        self,
        bias_type: str,
        bias_data: Any,
        cultural_context: CulturalContext
    ) -> BiasAdjustment:
        """Calculate cultural adjustment for a specific bias."""

        # Extract original severity
        if isinstance(bias_data, dict) and "severity" in bias_data:
            original_severity = float(bias_data["severity"])
        elif isinstance(bias_data, (int, float)):
            original_severity = float(bias_data)
        else:
            original_severity = 0.5  # Default moderate severity

        # Calculate cultural modifier based on dimension differences
        cultural_modifier = self._calculate_cultural_modifier(bias_type, cultural_context)

        # Apply modifier with bounds checking
        adjusted_severity = max(0.0, min(1.0, original_severity * cultural_modifier))

        # Generate explanation
        explanation = self._generate_adjustment_explanation(
            bias_type, cultural_modifier, cultural_context
        )

        # Identify cultural factors
        cultural_factors = self._identify_cultural_factors(bias_type, cultural_context)

        return BiasAdjustment(
            original_severity=original_severity,
            adjusted_severity=adjusted_severity,
            cultural_modifier=cultural_modifier,
            explanation=explanation,
            cultural_factors=cultural_factors
        )

    def _calculate_cultural_modifier(
        self,
        bias_type: str,
        cultural_context: CulturalContext
    ) -> float:
        """Calculate the cultural modifier for a specific bias type."""

        # Get bias type sensitivity weights
        sensitivities = self.BIAS_CULTURAL_SENSITIVITY.get(
            bias_type.lower(),
            self.BIAS_CULTURAL_SENSITIVITY["racial"]  # Default fallback
        )

        # Get dimension differences
        differences = self.hofstede_model.get_dimension_differences(
            cultural_context.sender_culture.dimensions,
            cultural_context.receiver_culture.dimensions
        )

        # Calculate weighted modifier
        total_weighted_diff = 0.0
        total_weight = 0.0

        for dim_code, sensitivity in sensitivities.items():
            if dim_code in differences:
                diff_data = differences[dim_code]
                normalized_diff = diff_data["percentage_difference"] / 100.0  # 0-1 scale
                weighted_diff = normalized_diff * sensitivity

                total_weighted_diff += weighted_diff
                total_weight += sensitivity

        # Calculate base modifier (higher differences = higher modifier)
        if total_weight > 0:
            avg_weighted_diff = total_weighted_diff / total_weight
            base_modifier = 1.0 + (avg_weighted_diff * 0.5)  # Max 1.5x increase
        else:
            base_modifier = 1.0

        # Apply risk level adjustments
        risk_adjustments = {
            "low": 0.95,
            "medium": 1.0,
            "high": 1.1,
            "very_high": 1.2
        }

        risk_modifier = risk_adjustments.get(cultural_context.risk_level, 1.0)
        final_modifier = base_modifier * risk_modifier

        # Apply bounds based on cultural distance
        distance = cultural_context.cultural_distance
        if distance <= 20:
            modifier_range = self.MODIFIER_THRESHOLDS["minimal"]
        elif distance <= 35:
            modifier_range = self.MODIFIER_THRESHOLDS["moderate"]
        elif distance <= 50:
            modifier_range = self.MODIFIER_THRESHOLDS["significant"]
        else:
            modifier_range = self.MODIFIER_THRESHOLDS["major"]

        # Clamp modifier within appropriate range
        final_modifier = max(
            modifier_range["min"],
            min(modifier_range["max"], final_modifier)
        )

        return final_modifier

    def _generate_adjustment_explanation(
        self,
        bias_type: str,
        modifier: float,
        cultural_context: CulturalContext
    ) -> str:
        """Generate human-readable explanation for cultural adjustment."""

        sender_country = cultural_context.sender_culture.country
        receiver_country = cultural_context.receiver_culture.country
        distance = cultural_context.cultural_distance

        if modifier > 1.1:
            intensity = "significantly amplified"
        elif modifier > 1.05:
            intensity = "moderately amplified"
        elif modifier < 0.9:
            intensity = "significantly reduced"
        elif modifier < 0.95:
            intensity = "moderately reduced"
        else:
            intensity = "minimally affected"

        explanation = (
            f"Bias severity {intensity} due to cultural differences between "
            f"{sender_country} and {receiver_country} (cultural distance: {distance:.1f}). "
        )

        # Add specific cultural factor explanations
        if cultural_context.high_risk_dimensions:
            risk_dims = [dim["dimension"] for dim in cultural_context.high_risk_dimensions]
            explanation += f"High-risk cultural dimensions: {', '.join(risk_dims)}."

        return explanation

    def _identify_cultural_factors(
        self,
        bias_type: str,
        cultural_context: CulturalContext
    ) -> List[str]:
        """Identify specific cultural factors affecting the bias."""

        factors = []

        # Add risk dimensions as factors
        for risk_dim in cultural_context.high_risk_dimensions:
            dim_name = risk_dim["name"]
            factors.append(f"Cultural difference in {dim_name}")

        # Add communication style factors
        sender_style = cultural_context.sender_culture.characteristics.get("communication_style", "unknown")
        receiver_style = cultural_context.receiver_culture.characteristics.get("communication_style", "unknown")

        if sender_style != receiver_style and sender_style != "unknown" and receiver_style != "unknown":
            factors.append(f"Communication style mismatch: {sender_style} vs {receiver_style}")

        # Add hierarchy factors for relevant biases
        if bias_type.lower() in ["age", "socioeconomic", "gender"]:
            sender_hierarchy = cultural_context.sender_culture.characteristics.get("hierarchy_acceptance", "medium")
            receiver_hierarchy = cultural_context.receiver_culture.characteristics.get("hierarchy_acceptance", "medium")

            if sender_hierarchy != receiver_hierarchy:
                factors.append(f"Different hierarchy expectations: {sender_hierarchy} vs {receiver_hierarchy}")

        return factors

    def _serialize_cultural_context(self, context: CulturalContext) -> Dict[str, Any]:
        """Serialize cultural context for inclusion in results."""
        return {
            "sender_culture": {
                "country": context.sender_culture.country,
                "code": context.sender_culture.code,
                "dimensions": context.sender_culture.dimensions.to_dict()
            },
            "receiver_culture": {
                "country": context.receiver_culture.country,
                "code": context.receiver_culture.code,
                "dimensions": context.receiver_culture.dimensions.to_dict()
            },
            "cultural_distance": context.cultural_distance,
            "risk_level": context.risk_level,
            "mitigation_strategies": context.mitigation_strategies
        }

    def _update_overall_scores(
        self,
        results: Dict[str, Any],
        cultural_context: CulturalContext
    ) -> None:
        """Update overall bias scores with cultural considerations."""

        if "overall_bias_score" in results:
            original_score = results["overall_bias_score"]

            # Calculate average cultural modifier from all adjustments
            if "cultural_adjustments" in results:
                modifiers = [
                    adj["cultural_modifier"]
                    for adj in results["cultural_adjustments"].values()
                ]
                if modifiers:
                    avg_modifier = sum(modifiers) / len(modifiers)
                    results["culturally_adjusted_score"] = min(1.0, original_score * avg_modifier)

        # Add cultural risk as separate metric
        results["cultural_communication_risk"] = {
            "level": cultural_context.risk_level,
            "distance": cultural_context.cultural_distance,
            "requires_mitigation": len(cultural_context.mitigation_strategies) > 0
        }

    def get_cultural_explanation(
        self,
        bias_type: str,
        sender_culture: str,
        receiver_culture: str
    ) -> Dict[str, Any]:
        """
        Get detailed cultural explanation for a bias type between cultures.

        Args:
            bias_type: Type of bias to analyze
            sender_culture: Sender's culture code
            receiver_culture: Receiver's culture code

        Returns:
            Detailed cultural explanation
        """
        try:
            sender_profile = self.profile_manager.get_profile(sender_culture)
            receiver_profile = self.profile_manager.get_profile(receiver_culture)
            cultural_context = self._analyze_cultural_context(sender_profile, receiver_profile)

            return {
                "bias_type": bias_type,
                "cultural_context": self._serialize_cultural_context(cultural_context),
                "sensitivity_factors": self.BIAS_CULTURAL_SENSITIVITY.get(bias_type.lower(), {}),
                "cultural_explanation": self._generate_detailed_explanation(
                    bias_type, cultural_context
                ),
                "mitigation_strategies": cultural_context.mitigation_strategies
            }

        except Exception as e:
            self.logger.error(f"Error generating cultural explanation: {e}")
            return {"error": str(e)}

    def _generate_detailed_explanation(
        self,
        bias_type: str,
        cultural_context: CulturalContext
    ) -> str:
        """Generate detailed explanation of cultural factors for a bias type."""

        sender_country = cultural_context.sender_culture.country
        receiver_country = cultural_context.receiver_culture.country

        explanation = (
            f"When analyzing {bias_type} bias between {sender_country} and {receiver_country}, "
            f"several cultural factors influence perception and severity:\n\n"
        )

        # Add dimension-specific explanations
        for risk_dim in cultural_context.high_risk_dimensions:
            dim_code = risk_dim["dimension"]
            explanation += f"• {risk_dim['name']}: {risk_dim['difference']:.0f} point difference\n"

        if cultural_context.mitigation_strategies:
            explanation += f"\nRecommended strategies:\n"
            for strategy in cultural_context.mitigation_strategies:
                explanation += f"• {strategy}\n"

        return explanation.strip()