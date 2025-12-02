"""
Cultural Intelligence System

Provides advanced cultural intelligence features including radar charts,
communication style analysis, and cross-cultural sensitivity recommendations.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import math

from ..models.hofstede_model import HofstedeModel, CulturalDimensions, CulturalProfile
from ..data.profile_manager import CulturalProfileManager


@dataclass
class CommunicationStyle:
    """Represents communication style characteristics for a culture."""
    directness: str  # direct, indirect, moderate
    formality: str   # formal, informal, balanced
    emotionality: str  # expressive, restrained, moderate
    context: str     # high_context, low_context, mixed
    hierarchy_sensitivity: str  # high, medium, low


@dataclass
class CulturalRadarData:
    """Data structure for cultural radar chart visualization."""
    culture_code: str
    culture_name: str
    dimensions: Dict[str, int]
    dimension_labels: List[str]
    color: str
    comparison_scores: Optional[Dict[str, float]] = None


class CulturalIntelligence:
    """
    Advanced cultural intelligence system providing comprehensive
    cultural analysis, visualization, and recommendation capabilities.
    """

    def __init__(self, profile_manager: Optional[CulturalProfileManager] = None):
        """Initialize the cultural intelligence system."""
        self.logger = logging.getLogger(__name__)
        self.profile_manager = profile_manager or CulturalProfileManager()
        self.hofstede_model = HofstedeModel()

        # Communication style mapping based on cultural dimensions
        self.COMMUNICATION_STYLE_MAPPING = {
            "directness": {
                "factors": {"PDI": -0.3, "UAI": -0.2, "IDV": 0.4},
                "thresholds": {"direct": 60, "indirect": 40}
            },
            "formality": {
                "factors": {"PDI": 0.5, "UAI": 0.3, "LTO": 0.2},
                "thresholds": {"formal": 65, "informal": 35}
            },
            "emotionality": {
                "factors": {"IVR": 0.6, "MAS": -0.3, "UAI": -0.2},
                "thresholds": {"expressive": 60, "restrained": 40}
            },
            "context": {
                "factors": {"IDV": -0.4, "PDI": 0.3, "UAI": 0.3},
                "thresholds": {"high_context": 55, "low_context": 45}
            }
        }

        # Cultural sensitivity warnings
        self.SENSITIVITY_WARNINGS = {
            "high_pdi_difference": {
                "threshold": 30,
                "warning": "Significant power distance difference - be mindful of hierarchy expectations",
                "severity": "high"
            },
            "high_uai_difference": {
                "threshold": 35,
                "warning": "Major uncertainty avoidance difference - provide clear structure when needed",
                "severity": "high"
            },
            "individualism_collectivism_clash": {
                "threshold": 40,
                "warning": "Individual vs collective orientation clash - balance personal and group perspectives",
                "severity": "medium"
            },
            "time_orientation_mismatch": {
                "threshold": 45,
                "warning": "Different time orientations - align planning and evaluation timeframes",
                "severity": "medium"
            }
        }

    def generate_radar_chart_data(
        self,
        culture_codes: List[str],
        include_comparisons: bool = False
    ) -> List[CulturalRadarData]:
        """
        Generate radar chart data for multiple cultures.

        Args:
            culture_codes: List of culture codes to include
            include_comparisons: Whether to include comparison scores

        Returns:
            List of radar chart data structures
        """
        radar_data = []

        for code in culture_codes:
            try:
                profile = self.profile_manager.get_profile(code)

                # Generate comparison scores if requested
                comparison_scores = None
                if include_comparisons and len(culture_codes) > 1:
                    comparison_scores = self._calculate_comparison_scores(
                        profile, [self.profile_manager.get_profile(c) for c in culture_codes if c != code]
                    )

                radar_data.append(CulturalRadarData(
                    culture_code=profile.code,
                    culture_name=profile.country,
                    dimensions=profile.dimensions.to_dict(),
                    dimension_labels=list(self.hofstede_model.DIMENSION_NAMES.values()),
                    color=self._get_culture_color(profile.code),
                    comparison_scores=comparison_scores
                ))

            except Exception as e:
                self.logger.error(f"Error generating radar data for {code}: {e}")
                continue

        return radar_data

    def _calculate_comparison_scores(
        self,
        target_profile: CulturalProfile,
        comparison_profiles: List[CulturalProfile]
    ) -> Dict[str, float]:
        """Calculate comparison scores against other cultures."""
        scores = {}

        for comparison_profile in comparison_profiles:
            distance = self.hofstede_model.calculate_cultural_distance(
                target_profile.dimensions,
                comparison_profile.dimensions
            )
            similarity_score = max(0, 100 - distance)
            scores[comparison_profile.code] = similarity_score

        return scores

    def _get_culture_color(self, culture_code: str) -> str:
        """Get consistent color for a culture."""
        color_palette = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57",
            "#FF9FF3", "#54A0FF", "#5F27CD", "#00D2D3", "#FF9F43",
            "#26de81", "#a55eea", "#fd79a8", "#e17055", "#6c5ce7"
        ]

        # Generate consistent color based on culture code hash
        hash_value = hash(culture_code) % len(color_palette)
        return color_palette[hash_value]

    def analyze_communication_style(self, culture_code: str) -> CommunicationStyle:
        """
        Analyze communication style for a culture based on dimensions.

        Args:
            culture_code: Culture code to analyze

        Returns:
            Communication style analysis
        """
        profile = self.profile_manager.get_profile(culture_code)
        dimensions = profile.dimensions.to_dict()

        style = CommunicationStyle(
            directness=self._calculate_style_attribute("directness", dimensions),
            formality=self._calculate_style_attribute("formality", dimensions),
            emotionality=self._calculate_style_attribute("emotionality", dimensions),
            context=self._calculate_style_attribute("context", dimensions),
            hierarchy_sensitivity=self._calculate_hierarchy_sensitivity(dimensions["PDI"])
        )

        return style

    def _calculate_style_attribute(self, attribute: str, dimensions: Dict[str, int]) -> str:
        """Calculate a specific style attribute based on dimensions."""
        if attribute not in self.COMMUNICATION_STYLE_MAPPING:
            return "moderate"

        mapping = self.COMMUNICATION_STYLE_MAPPING[attribute]
        factors = mapping["factors"]
        thresholds = mapping["thresholds"]

        # Calculate weighted score
        score = 50  # Base score
        for dim, weight in factors.items():
            score += (dimensions[dim] - 50) * weight

        # Apply thresholds
        high_threshold = max(thresholds.values())
        low_threshold = min(thresholds.values())

        if score >= high_threshold:
            return max(thresholds.keys(), key=lambda k: thresholds[k])
        elif score <= low_threshold:
            return min(thresholds.keys(), key=lambda k: thresholds[k])
        else:
            return "moderate" if "moderate" in ["direct", "indirect"] else "balanced"

    def _calculate_hierarchy_sensitivity(self, pdi_score: int) -> str:
        """Calculate hierarchy sensitivity level."""
        if pdi_score >= 70:
            return "high"
        elif pdi_score <= 30:
            return "low"
        else:
            return "medium"

    def generate_sensitivity_warnings(
        self,
        sender_culture: str,
        receiver_culture: str
    ) -> List[Dict[str, Any]]:
        """
        Generate cultural sensitivity warnings for cross-cultural communication.

        Args:
            sender_culture: Sender's culture code
            receiver_culture: Receiver's culture code

        Returns:
            List of sensitivity warnings
        """
        warnings = []

        try:
            sender_profile = self.profile_manager.get_profile(sender_culture)
            receiver_profile = self.profile_manager.get_profile(receiver_culture)

            differences = self.hofstede_model.get_dimension_differences(
                sender_profile.dimensions,
                receiver_profile.dimensions
            )

            # Check each sensitivity warning condition
            for warning_key, config in self.SENSITIVITY_WARNINGS.items():
                threshold = config["threshold"]
                warning_message = config["warning"]
                severity = config["severity"]

                triggered = False

                if warning_key == "high_pdi_difference":
                    if differences["PDI"]["absolute_difference"] >= threshold:
                        triggered = True

                elif warning_key == "high_uai_difference":
                    if differences["UAI"]["absolute_difference"] >= threshold:
                        triggered = True

                elif warning_key == "individualism_collectivism_clash":
                    if differences["IDV"]["absolute_difference"] >= threshold:
                        triggered = True

                elif warning_key == "time_orientation_mismatch":
                    if differences["LTO"]["absolute_difference"] >= threshold:
                        triggered = True

                if triggered:
                    warnings.append({
                        "type": warning_key,
                        "message": warning_message,
                        "severity": severity,
                        "dimensions_affected": self._get_affected_dimensions(warning_key),
                        "mitigation_strategies": self._get_mitigation_strategies(warning_key)
                    })

        except Exception as e:
            self.logger.error(f"Error generating sensitivity warnings: {e}")

        return warnings

    def _get_affected_dimensions(self, warning_type: str) -> List[str]:
        """Get dimensions affected by a warning type."""
        dimension_mapping = {
            "high_pdi_difference": ["PDI"],
            "high_uai_difference": ["UAI"],
            "individualism_collectivism_clash": ["IDV"],
            "time_orientation_mismatch": ["LTO"]
        }
        return dimension_mapping.get(warning_type, [])

    def _get_mitigation_strategies(self, warning_type: str) -> List[str]:
        """Get mitigation strategies for a warning type."""
        strategies = {
            "high_pdi_difference": [
                "Clarify authority structures explicitly",
                "Use appropriate levels of formality",
                "Be mindful of status and hierarchy cues"
            ],
            "high_uai_difference": [
                "Provide detailed information and structure",
                "Minimize ambiguity in communication",
                "Allow time for clarification and questions"
            ],
            "individualism_collectivism_clash": [
                "Balance individual and group recognition",
                "Consider collective impact of decisions",
                "Respect different responsibility attributions"
            ],
            "time_orientation_mismatch": [
                "Acknowledge different planning horizons",
                "Balance short-term and long-term perspectives",
                "Respect traditional vs innovative approaches"
            ]
        }
        return strategies.get(warning_type, [])

    def recommend_communication_strategies(
        self,
        sender_culture: str,
        receiver_culture: str,
        communication_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Recommend communication strategies for cross-cultural interaction.

        Args:
            sender_culture: Sender's culture code
            receiver_culture: Receiver's culture code
            communication_context: Optional context (business, educational, etc.)

        Returns:
            Comprehensive communication strategy recommendations
        """
        sender_profile = self.profile_manager.get_profile(sender_culture)
        receiver_profile = self.profile_manager.get_profile(receiver_culture)

        sender_style = self.analyze_communication_style(sender_culture)
        receiver_style = self.analyze_communication_style(receiver_culture)

        recommendations = {
            "sender_adaptations": self._generate_sender_adaptations(sender_style, receiver_style),
            "receiver_considerations": self._generate_receiver_considerations(sender_style, receiver_style),
            "general_strategies": self._generate_general_strategies(sender_profile, receiver_profile),
            "context_specific": self._generate_context_strategies(
                communication_context, sender_profile, receiver_profile
            ),
            "cultural_bridge_tips": self._generate_bridge_tips(sender_profile, receiver_profile)
        }

        return recommendations

    def _generate_sender_adaptations(
        self,
        sender_style: CommunicationStyle,
        receiver_style: CommunicationStyle
    ) -> List[str]:
        """Generate adaptations the sender should make."""
        adaptations = []

        # Directness adaptations
        if sender_style.directness == "direct" and receiver_style.directness == "indirect":
            adaptations.append("Use more diplomatic language and allow for face-saving")

        elif sender_style.directness == "indirect" and receiver_style.directness == "direct":
            adaptations.append("Be more explicit and specific in communication")

        # Formality adaptations
        if sender_style.formality == "informal" and receiver_style.formality == "formal":
            adaptations.append("Increase formality and use appropriate titles")

        elif sender_style.formality == "formal" and receiver_style.formality == "informal":
            adaptations.append("Adopt a more casual, approachable communication style")

        # Emotionality adaptations
        if sender_style.emotionality == "expressive" and receiver_style.emotionality == "restrained":
            adaptations.append("Moderate emotional expression and maintain composure")

        elif sender_style.emotionality == "restrained" and receiver_style.emotionality == "expressive":
            adaptations.append("Be more open with emotions and enthusiasm")

        return adaptations

    def _generate_receiver_considerations(
        self,
        sender_style: CommunicationStyle,
        receiver_style: CommunicationStyle
    ) -> List[str]:
        """Generate considerations for the receiver to understand the sender."""
        considerations = []

        if sender_style.directness != receiver_style.directness:
            considerations.append(f"Sender has {sender_style.directness} communication style")

        if sender_style.hierarchy_sensitivity != receiver_style.hierarchy_sensitivity:
            considerations.append(f"Sender has {sender_style.hierarchy_sensitivity} hierarchy sensitivity")

        if sender_style.context != receiver_style.context:
            considerations.append(f"Sender uses {sender_style.context.replace('_', ' ')} communication")

        return considerations

    def _generate_general_strategies(
        self,
        sender_profile: CulturalProfile,
        receiver_profile: CulturalProfile
    ) -> List[str]:
        """Generate general cross-cultural strategies."""
        strategies = [
            "Practice active listening and confirm understanding",
            "Be patient with cultural differences and misunderstandings",
            "Ask clarifying questions when in doubt",
            "Show respect for cultural backgrounds explicitly"
        ]

        # Add strategies based on cultural distance
        distance = self.hofstede_model.calculate_cultural_distance(
            sender_profile.dimensions,
            receiver_profile.dimensions
        )

        if distance > 50:
            strategies.extend([
                "Allow extra time for communication and relationship building",
                "Consider using cultural mediators or interpreters",
                "Document important agreements and decisions clearly"
            ])

        return strategies

    def _generate_context_strategies(
        self,
        context: Optional[str],
        sender_profile: CulturalProfile,
        receiver_profile: CulturalProfile
    ) -> List[str]:
        """Generate context-specific strategies."""
        if not context:
            return []

        strategies = []

        if context.lower() == "business":
            strategies.extend([
                "Establish clear meeting protocols and agendas",
                "Respect different approaches to decision-making timelines",
                "Be mindful of business hierarchy and authority structures"
            ])

        elif context.lower() == "educational":
            strategies.extend([
                "Adapt teaching/learning styles to cultural preferences",
                "Encourage questions while respecting cultural communication norms",
                "Provide multiple ways to demonstrate understanding"
            ])

        elif context.lower() == "healthcare":
            strategies.extend([
                "Respect cultural attitudes toward authority and medical decisions",
                "Be sensitive to cultural concepts of health and illness",
                "Consider family dynamics in medical discussions"
            ])

        return strategies

    def _generate_bridge_tips(
        self,
        sender_profile: CulturalProfile,
        receiver_profile: CulturalProfile
    ) -> List[str]:
        """Generate cultural bridge-building tips."""
        tips = [
            "Find common ground and shared values",
            "Learn about each other's cultural backgrounds",
            "Celebrate cultural diversity as a strength",
            "Create inclusive environments for both cultures"
        ]

        # Add dimension-specific bridge tips
        differences = self.hofstede_model.get_dimension_differences(
            sender_profile.dimensions,
            receiver_profile.dimensions
        )

        for dim_code, diff_data in differences.items():
            if diff_data["impact_level"] in ["significant", "major"]:
                bridge_tip = self._get_dimension_bridge_tip(dim_code)
                if bridge_tip:
                    tips.append(bridge_tip)

        return tips

    def _get_dimension_bridge_tip(self, dimension: str) -> Optional[str]:
        """Get bridge-building tip for a specific dimension."""
        bridge_tips = {
            "PDI": "Create hybrid hierarchy structures that respect both formal and informal authority",
            "IDV": "Design activities that honor both individual achievements and collective success",
            "MAS": "Balance competitive goals with collaborative relationship building",
            "UAI": "Provide structured frameworks while maintaining flexibility for adaptation",
            "LTO": "Integrate both traditional wisdom and innovative approaches",
            "IVR": "Create space for both expressive and restrained communication styles"
        }
        return bridge_tips.get(dimension)

    def generate_cultural_dashboard_data(
        self,
        culture_codes: List[str]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive dashboard data for cultural analysis.

        Args:
            culture_codes: List of culture codes to analyze

        Returns:
            Dashboard data structure
        """
        dashboard_data = {
            "cultures": [],
            "radar_chart": self.generate_radar_chart_data(culture_codes, include_comparisons=True),
            "cultural_distances": {},
            "communication_styles": {},
            "sensitivity_matrix": {},
            "recommendations": {}
        }

        # Generate data for each culture
        for code in culture_codes:
            profile = self.profile_manager.get_profile(code)
            dashboard_data["cultures"].append({
                "code": profile.code,
                "name": profile.country,
                "dimensions": profile.dimensions.to_dict(),
                "characteristics": profile.characteristics
            })

            # Communication style
            dashboard_data["communication_styles"][code] = self.analyze_communication_style(code).__dict__

        # Generate pairwise cultural distances
        for i, code1 in enumerate(culture_codes):
            for code2 in culture_codes[i+1:]:
                profile1 = self.profile_manager.get_profile(code1)
                profile2 = self.profile_manager.get_profile(code2)

                distance = self.hofstede_model.calculate_cultural_distance(
                    profile1.dimensions,
                    profile2.dimensions
                )

                dashboard_data["cultural_distances"][f"{code1}-{code2}"] = distance

                # Sensitivity warnings
                warnings = self.generate_sensitivity_warnings(code1, code2)
                dashboard_data["sensitivity_matrix"][f"{code1}-{code2}"] = warnings

                # Recommendations
                recommendations = self.recommend_communication_strategies(code1, code2)
                dashboard_data["recommendations"][f"{code1}-{code2}"] = recommendations

        return dashboard_data