"""
Cultural Context Analysis System

Provides comprehensive analysis of cross-cultural communication dynamics
and their impact on bias perception and interpretation.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..models.hofstede_model import HofstedeModel, CulturalDimensions, CulturalProfile
from ..data.profile_manager import CulturalProfileManager


class CommunicationRisk(Enum):
    """Enumeration of communication risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CulturalInsight:
    """Represents a specific cultural insight or observation."""
    dimension: str
    insight_type: str
    severity: str
    description: str
    impact: str
    mitigation: str


@dataclass
class CrossCulturalAnalysis:
    """Comprehensive cross-cultural analysis result."""
    sender_culture: str
    receiver_culture: str
    overall_risk: CommunicationRisk
    cultural_distance: float
    bridge_score: float
    insights: List[CulturalInsight]
    recommendations: List[str]
    potential_misunderstandings: List[str]


class CulturalAnalyzer:
    """
    Advanced cultural context analyzer for bias detection enhancement.

    Provides deep analysis of cultural dynamics, communication patterns,
    and potential areas of misunderstanding in cross-cultural scenarios.
    """

    def __init__(self, profile_manager: Optional[CulturalProfileManager] = None):
        """Initialize the cultural analyzer."""
        self.logger = logging.getLogger(__name__)
        self.profile_manager = profile_manager or CulturalProfileManager()
        self.hofstede_model = HofstedeModel()

        # Cultural insight patterns
        self.INSIGHT_PATTERNS = {
            "PDI": {
                "high_diff": {
                    "insight_type": "hierarchy_mismatch",
                    "description": "Significant difference in power distance expectations",
                    "impact": "May lead to misinterpretation of authority and respect cues",
                    "mitigation": "Clarify hierarchical relationships and decision-making processes"
                },
                "cultural_contexts": {
                    "high_to_low": "High hierarchy culture communicating with low hierarchy culture",
                    "low_to_high": "Low hierarchy culture communicating with high hierarchy culture"
                }
            },
            "IDV": {
                "high_diff": {
                    "insight_type": "collective_individual_tension",
                    "description": "Major divergence in individual vs collective orientation",
                    "impact": "May cause misunderstanding of group vs personal responsibility",
                    "mitigation": "Balance individual achievements with group considerations"
                },
                "cultural_contexts": {
                    "individual_to_collective": "Individualistic culture communicating with collective culture",
                    "collective_to_individual": "Collective culture communicating with individualistic culture"
                }
            },
            "MAS": {
                "high_diff": {
                    "insight_type": "achievement_relationship_gap",
                    "description": "Different emphasis on achievement vs relationship building",
                    "impact": "May lead to misaligned priorities and communication styles",
                    "mitigation": "Acknowledge both task and relationship aspects"
                }
            },
            "UAI": {
                "high_diff": {
                    "insight_type": "uncertainty_comfort_mismatch",
                    "description": "Contrasting tolerance for ambiguity and uncertainty",
                    "impact": "May cause stress and misunderstanding in ambiguous situations",
                    "mitigation": "Provide structure and clarity when addressing uncertainty-averse cultures"
                }
            },
            "LTO": {
                "high_diff": {
                    "insight_type": "time_orientation_conflict",
                    "description": "Different perspectives on time, tradition, and change",
                    "impact": "May lead to misaligned expectations about planning and results",
                    "mitigation": "Acknowledge different time perspectives in planning and evaluation"
                }
            },
            "IVR": {
                "high_diff": {
                    "insight_type": "expression_restraint_gap",
                    "description": "Contrasting approaches to emotional expression and gratification",
                    "impact": "May cause misinterpretation of emotional cues and social behavior",
                    "mitigation": "Respect different expression styles and social norms"
                }
            }
        }

    def analyze_cross_cultural_context(
        self,
        sender_culture: str,
        receiver_culture: str,
        communication_context: Optional[Dict[str, Any]] = None
    ) -> CrossCulturalAnalysis:
        """
        Perform comprehensive cross-cultural analysis.

        Args:
            sender_culture: Sender's culture code
            receiver_culture: Receiver's culture code
            communication_context: Optional context information

        Returns:
            Comprehensive cross-cultural analysis
        """
        try:
            # Get cultural profiles
            sender_profile = self.profile_manager.get_profile(sender_culture)
            receiver_profile = self.profile_manager.get_profile(receiver_culture)

            # Calculate basic metrics
            risk_assessment = self.hofstede_model.assess_communication_risk(
                sender_profile.dimensions,
                receiver_profile.dimensions
            )

            cultural_distance = risk_assessment["overall_distance"]
            bridge_score = risk_assessment["cultural_bridge_score"]

            # Generate cultural insights
            insights = self._generate_cultural_insights(sender_profile, receiver_profile)

            # Determine overall risk level
            overall_risk = self._determine_risk_level(cultural_distance, insights)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                sender_profile, receiver_profile, insights, communication_context
            )

            # Identify potential misunderstandings
            misunderstandings = self._identify_potential_misunderstandings(
                sender_profile, receiver_profile, insights
            )

            return CrossCulturalAnalysis(
                sender_culture=sender_culture,
                receiver_culture=receiver_culture,
                overall_risk=overall_risk,
                cultural_distance=cultural_distance,
                bridge_score=bridge_score,
                insights=insights,
                recommendations=recommendations,
                potential_misunderstandings=misunderstandings
            )

        except Exception as e:
            self.logger.error(f"Error in cross-cultural analysis: {e}")
            raise

    def _generate_cultural_insights(
        self,
        sender_profile: CulturalProfile,
        receiver_profile: CulturalProfile
    ) -> List[CulturalInsight]:
        """Generate specific cultural insights based on dimension differences."""
        insights = []

        # Get dimension differences
        differences = self.hofstede_model.get_dimension_differences(
            sender_profile.dimensions,
            receiver_profile.dimensions
        )

        # Generate insights for each significant difference
        for dim_code, diff_data in differences.items():
            if diff_data["impact_level"] in ["significant", "major"]:
                insight = self._create_dimension_insight(
                    dim_code, diff_data, sender_profile, receiver_profile
                )
                if insight:
                    insights.append(insight)

        return insights

    def _create_dimension_insight(
        self,
        dimension: str,
        diff_data: Dict[str, Any],
        sender_profile: CulturalProfile,
        receiver_profile: CulturalProfile
    ) -> Optional[CulturalInsight]:
        """Create a cultural insight for a specific dimension difference."""

        if dimension not in self.INSIGHT_PATTERNS:
            return None

        pattern = self.INSIGHT_PATTERNS[dimension]["high_diff"]

        # Determine severity based on difference magnitude
        severity = diff_data["impact_level"]

        # Create detailed description
        description = f"{pattern['description']} ({diff_data['absolute_difference']:.0f} points difference)"

        # Add cultural direction context if available
        sender_value = diff_data["culture1_value"]
        receiver_value = diff_data["culture2_value"]

        if dimension in ["PDI", "UAI", "LTO"] and abs(sender_value - receiver_value) > 20:
            if sender_value > receiver_value:
                direction = "high_to_low"
            else:
                direction = "low_to_high"

            context_patterns = self.INSIGHT_PATTERNS[dimension].get("cultural_contexts", {})
            if direction in context_patterns:
                description += f". {context_patterns[direction]}."

        return CulturalInsight(
            dimension=dimension,
            insight_type=pattern["insight_type"],
            severity=severity,
            description=description,
            impact=pattern["impact"],
            mitigation=pattern["mitigation"]
        )

    def _determine_risk_level(
        self,
        cultural_distance: float,
        insights: List[CulturalInsight]
    ) -> CommunicationRisk:
        """Determine overall communication risk level."""

        # Count high-severity insights
        critical_insights = len([i for i in insights if i.severity == "major"])
        significant_insights = len([i for i in insights if i.severity == "significant"])

        # Risk matrix based on distance and insights
        if cultural_distance <= 25 and critical_insights == 0:
            return CommunicationRisk.LOW
        elif cultural_distance <= 40 and critical_insights <= 1:
            return CommunicationRisk.MEDIUM
        elif cultural_distance <= 60 and critical_insights <= 2:
            return CommunicationRisk.HIGH
        else:
            return CommunicationRisk.CRITICAL

    def _generate_recommendations(
        self,
        sender_profile: CulturalProfile,
        receiver_profile: CulturalProfile,
        insights: List[CulturalInsight],
        communication_context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Generate specific recommendations for cross-cultural communication."""

        recommendations = []

        # Add insight-specific recommendations
        for insight in insights:
            if insight.mitigation and insight.mitigation not in recommendations:
                recommendations.append(insight.mitigation)

        # Add context-specific recommendations
        if communication_context:
            context_type = communication_context.get("type", "general")

            if context_type == "business":
                recommendations.extend(self._get_business_recommendations(sender_profile, receiver_profile))
            elif context_type == "educational":
                recommendations.extend(self._get_educational_recommendations(sender_profile, receiver_profile))

        # Add general cross-cultural recommendations
        recommendations.extend([
            "Use clear, specific language and avoid idioms",
            "Allow extra time for clarification and questions",
            "Be aware of non-verbal communication differences",
            "Show respect for cultural differences explicitly"
        ])

        return list(set(recommendations))  # Remove duplicates

    def _get_business_recommendations(
        self,
        sender_profile: CulturalProfile,
        receiver_profile: CulturalProfile
    ) -> List[str]:
        """Generate business-specific recommendations."""
        recommendations = []

        # Power distance considerations
        pdi_diff = abs(sender_profile.dimensions.PDI - receiver_profile.dimensions.PDI)
        if pdi_diff > 20:
            recommendations.append("Clarify decision-making authority and reporting structures")

        # Individualism considerations
        idv_diff = abs(sender_profile.dimensions.IDV - receiver_profile.dimensions.IDV)
        if idv_diff > 25:
            recommendations.append("Balance individual and team recognition appropriately")

        # Uncertainty avoidance considerations
        uai_diff = abs(sender_profile.dimensions.UAI - receiver_profile.dimensions.UAI)
        if uai_diff > 30:
            recommendations.append("Provide detailed project plans and risk assessments")

        return recommendations

    def _get_educational_recommendations(
        self,
        sender_profile: CulturalProfile,
        receiver_profile: CulturalProfile
    ) -> List[str]:
        """Generate education-specific recommendations."""
        recommendations = []

        # Power distance in educational context
        pdi_diff = abs(sender_profile.dimensions.PDI - receiver_profile.dimensions.PDI)
        if pdi_diff > 25:
            recommendations.append("Establish clear teacher-student relationship expectations")

        # Individualism in learning
        idv_diff = abs(sender_profile.dimensions.IDV - receiver_profile.dimensions.IDV)
        if idv_diff > 30:
            recommendations.append("Provide both individual and group learning opportunities")

        return recommendations

    def _identify_potential_misunderstandings(
        self,
        sender_profile: CulturalProfile,
        receiver_profile: CulturalProfile,
        insights: List[CulturalInsight]
    ) -> List[str]:
        """Identify potential areas of misunderstanding."""
        misunderstandings = []

        for insight in insights:
            if insight.dimension == "PDI" and insight.severity in ["significant", "major"]:
                misunderstandings.append("Authority and hierarchy may be perceived differently")

            elif insight.dimension == "IDV" and insight.severity in ["significant", "major"]:
                misunderstandings.append("Individual vs group responsibility expectations may clash")

            elif insight.dimension == "UAI" and insight.severity in ["significant", "major"]:
                misunderstandings.append("Comfort with ambiguity may differ significantly")

            elif insight.dimension == "LTO" and insight.severity in ["significant", "major"]:
                misunderstandings.append("Time orientation and planning perspectives may differ")

        # Add communication style misunderstandings
        sender_style = sender_profile.characteristics.get("communication_style", "neutral")
        receiver_style = receiver_profile.characteristics.get("communication_style", "neutral")

        if sender_style != receiver_style:
            style_map = {
                ("direct", "indirect"): "Direct communication may be perceived as rude or aggressive",
                ("indirect", "direct"): "Indirect communication may be perceived as evasive or unclear",
                ("formal", "assertive"): "Formal communication may be seen as distant or cold",
                ("assertive", "formal"): "Assertive communication may be seen as disrespectful"
            }

            key = (sender_style, receiver_style)
            if key in style_map:
                misunderstandings.append(style_map[key])

        return misunderstandings

    def assess_communication_risk(
        self,
        cultural_context: Dict[str, Any],
        bias_severity: float
    ) -> Dict[str, Any]:
        """
        Assess overall communication risk given cultural context and bias severity.

        Args:
            cultural_context: Cultural context information
            bias_severity: Original bias severity score

        Returns:
            Risk assessment with recommendations
        """
        risk_level = cultural_context.get("risk_level", "medium")
        cultural_distance = cultural_context.get("cultural_distance", 0)

        # Calculate combined risk score
        cultural_risk_weight = {
            "low": 0.1,
            "medium": 0.3,
            "high": 0.5,
            "very_high": 0.7
        }.get(risk_level, 0.3)

        combined_risk = min(1.0, bias_severity + (cultural_risk_weight * (cultural_distance / 100)))

        # Generate risk assessment
        assessment = {
            "combined_risk_score": combined_risk,
            "cultural_risk_contribution": cultural_risk_weight * (cultural_distance / 100),
            "bias_risk_contribution": bias_severity,
            "risk_level": self._categorize_combined_risk(combined_risk),
            "requires_intervention": combined_risk > 0.6,
            "monitoring_priority": self._get_monitoring_priority(combined_risk, cultural_distance)
        }

        return assessment

    def _categorize_combined_risk(self, combined_risk: float) -> str:
        """Categorize combined risk score."""
        if combined_risk <= 0.3:
            return "low"
        elif combined_risk <= 0.5:
            return "moderate"
        elif combined_risk <= 0.7:
            return "high"
        else:
            return "critical"

    def _get_monitoring_priority(self, combined_risk: float, cultural_distance: float) -> str:
        """Determine monitoring priority level."""
        if combined_risk > 0.7 or cultural_distance > 60:
            return "high"
        elif combined_risk > 0.5 or cultural_distance > 40:
            return "medium"
        else:
            return "low"

    def generate_cultural_bridge_suggestions(
        self,
        sender_culture: str,
        receiver_culture: str
    ) -> List[str]:
        """Generate specific suggestions for building cultural bridges."""

        sender_profile = self.profile_manager.get_profile(sender_culture)
        receiver_profile = self.profile_manager.get_profile(receiver_culture)

        suggestions = []

        # Analyze each dimension for bridge-building opportunities
        differences = self.hofstede_model.get_dimension_differences(
            sender_profile.dimensions,
            receiver_profile.dimensions
        )

        for dim_code, diff_data in differences.items():
            if diff_data["impact_level"] in ["moderate", "significant", "major"]:
                bridge_suggestion = self._get_bridge_suggestion(dim_code, diff_data)
                if bridge_suggestion:
                    suggestions.append(bridge_suggestion)

        return suggestions

    def _get_bridge_suggestion(self, dimension: str, diff_data: Dict[str, Any]) -> Optional[str]:
        """Get bridge-building suggestion for a specific dimension."""

        bridge_strategies = {
            "PDI": "Find middle ground in formality and acknowledge both hierarchical and egalitarian perspectives",
            "IDV": "Balance individual recognition with team achievements and collective goals",
            "MAS": "Acknowledge both competitive achievements and collaborative relationship building",
            "UAI": "Provide structure for uncertainty-averse cultures while maintaining flexibility",
            "LTO": "Balance short-term results with long-term vision and planning",
            "IVR": "Respect both expressive and restrained communication styles"
        }

        return bridge_strategies.get(dimension)