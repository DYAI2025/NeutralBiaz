"""
Hofstede Cultural Dimensions Model

Implements the 6D cultural framework for analyzing cultural differences
and their impact on communication and bias perception.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import json
from pathlib import Path


@dataclass
class CulturalDimensions:
    """Represents Hofstede's 6 cultural dimensions for a country."""

    PDI: int  # Power Distance Index
    IDV: int  # Individualism vs Collectivism
    MAS: int  # Masculinity vs Femininity
    UAI: int  # Uncertainty Avoidance Index
    LTO: int  # Long Term Orientation vs Short Term
    IVR: int  # Indulgence vs Restraint

    def to_dict(self) -> Dict[str, int]:
        """Convert dimensions to dictionary format."""
        return {
            "PDI": self.PDI,
            "IDV": self.IDV,
            "MAS": self.MAS,
            "UAI": self.UAI,
            "LTO": self.LTO,
            "IVR": self.IVR
        }

    def to_list(self) -> List[int]:
        """Convert dimensions to list format for calculations."""
        return [self.PDI, self.IDV, self.MAS, self.UAI, self.LTO, self.IVR]

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'CulturalDimensions':
        """Create CulturalDimensions from dictionary."""
        return cls(
            PDI=data["PDI"],
            IDV=data["IDV"],
            MAS=data["MAS"],
            UAI=data["UAI"],
            LTO=data["LTO"],
            IVR=data["IVR"]
        )


@dataclass
class CulturalProfile:
    """Complete cultural profile for a country including dimensions and characteristics."""

    country: str
    code: str
    dimensions: CulturalDimensions
    characteristics: Dict[str, str]

    @classmethod
    def from_dict(cls, data: Dict) -> 'CulturalProfile':
        """Create CulturalProfile from dictionary data."""
        return cls(
            country=data["country"],
            code=data["code"],
            dimensions=CulturalDimensions.from_dict(data["dimensions"]),
            characteristics=data["characteristics"]
        )


class HofstedeModel:
    """
    Core implementation of Hofstede's Cultural Dimensions Model.

    Provides methods for cultural distance calculation, dimension analysis,
    and cultural context assessment for bias detection enhancement.
    """

    DIMENSION_NAMES = {
        "PDI": "Power Distance Index",
        "IDV": "Individualism vs Collectivism",
        "MAS": "Masculinity vs Femininity",
        "UAI": "Uncertainty Avoidance Index",
        "LTO": "Long Term Orientation vs Short Term",
        "IVR": "Indulgence vs Restraint"
    }

    DIMENSION_WEIGHTS = {
        "PDI": 1.2,  # Higher weight for power distance in communication
        "IDV": 1.0,  # Standard weight for individualism
        "MAS": 0.8,  # Lower weight for masculinity in bias context
        "UAI": 1.3,  # Higher weight for uncertainty avoidance
        "LTO": 0.9,  # Moderate weight for time orientation
        "IVR": 1.1   # Moderate weight for indulgence
    }

    def __init__(self):
        """Initialize the Hofstede model."""
        self.profiles: Dict[str, CulturalProfile] = {}
        self.default_profile: Optional[CulturalProfile] = None

    def calculate_cultural_distance(
        self,
        culture1: CulturalDimensions,
        culture2: CulturalDimensions,
        use_weights: bool = True
    ) -> float:
        """
        Calculate the cultural distance between two cultures using weighted Euclidean distance.

        Args:
            culture1: First cultural dimensions
            culture2: Second cultural dimensions
            use_weights: Whether to apply dimension weights

        Returns:
            Cultural distance value (0-100 scale)
        """
        dims1 = culture1.to_list()
        dims2 = culture2.to_list()
        weights = list(self.DIMENSION_WEIGHTS.values()) if use_weights else [1.0] * 6

        # Calculate weighted squared differences
        squared_diffs = []
        for i, (d1, d2, weight) in enumerate(zip(dims1, dims2, weights)):
            squared_diff = weight * (d1 - d2) ** 2
            squared_diffs.append(squared_diff)

        # Calculate distance and normalize to 0-100 scale
        distance = math.sqrt(sum(squared_diffs))
        max_possible_distance = math.sqrt(sum(w * 100**2 for w in weights))
        normalized_distance = (distance / max_possible_distance) * 100

        return min(normalized_distance, 100.0)

    def get_dimension_differences(
        self,
        culture1: CulturalDimensions,
        culture2: CulturalDimensions
    ) -> Dict[str, Dict[str, float]]:
        """
        Get detailed differences for each cultural dimension.

        Args:
            culture1: First cultural dimensions
            culture2: Second cultural dimensions

        Returns:
            Dictionary with dimension differences and analysis
        """
        dims1 = culture1.to_dict()
        dims2 = culture2.to_dict()

        differences = {}
        for dim_code in self.DIMENSION_NAMES:
            diff = abs(dims1[dim_code] - dims2[dim_code])
            percentage_diff = (diff / 100) * 100

            differences[dim_code] = {
                "name": self.DIMENSION_NAMES[dim_code],
                "culture1_value": dims1[dim_code],
                "culture2_value": dims2[dim_code],
                "absolute_difference": diff,
                "percentage_difference": percentage_diff,
                "impact_level": self._categorize_difference(diff),
                "weight": self.DIMENSION_WEIGHTS[dim_code]
            }

        return differences

    def _categorize_difference(self, difference: float) -> str:
        """Categorize the impact level of a cultural difference."""
        if difference <= 10:
            return "minimal"
        elif difference <= 25:
            return "moderate"
        elif difference <= 40:
            return "significant"
        else:
            return "major"

    def assess_communication_risk(
        self,
        sender_culture: CulturalDimensions,
        receiver_culture: CulturalDimensions
    ) -> Dict[str, any]:
        """
        Assess communication risks based on cultural differences.

        Args:
            sender_culture: Sender's cultural dimensions
            receiver_culture: Receiver's cultural dimensions

        Returns:
            Comprehensive risk assessment
        """
        distance = self.calculate_cultural_distance(sender_culture, receiver_culture)
        differences = self.get_dimension_differences(sender_culture, receiver_culture)

        # Identify high-risk dimensions
        high_risk_dimensions = []
        for dim_code, diff_data in differences.items():
            if diff_data["impact_level"] in ["significant", "major"]:
                high_risk_dimensions.append({
                    "dimension": dim_code,
                    "name": diff_data["name"],
                    "difference": diff_data["absolute_difference"],
                    "risk_factors": self._get_risk_factors(dim_code, diff_data)
                })

        # Calculate overall risk level
        risk_level = self._calculate_risk_level(distance, len(high_risk_dimensions))

        return {
            "overall_distance": distance,
            "risk_level": risk_level,
            "high_risk_dimensions": high_risk_dimensions,
            "communication_challenges": self._get_communication_challenges(differences),
            "mitigation_strategies": self._get_mitigation_strategies(high_risk_dimensions),
            "cultural_bridge_score": 100 - distance
        }

    def _get_risk_factors(self, dimension: str, diff_data: Dict) -> List[str]:
        """Get specific risk factors for a cultural dimension difference."""
        risk_factors = {
            "PDI": [
                "Hierarchy expectations mismatch",
                "Authority perception differences",
                "Decision-making process conflicts"
            ],
            "IDV": [
                "Individual vs group focus conflicts",
                "Responsibility attribution differences",
                "Personal space and autonomy expectations"
            ],
            "MAS": [
                "Competition vs cooperation preferences",
                "Achievement vs relationship focus",
                "Gender role expectation differences"
            ],
            "UAI": [
                "Risk tolerance differences",
                "Structure vs flexibility preferences",
                "Ambiguity handling conflicts"
            ],
            "LTO": [
                "Time perspective misalignment",
                "Tradition vs innovation conflicts",
                "Planning horizon differences"
            ],
            "IVR": [
                "Expression vs restraint differences",
                "Social norm expectations",
                "Gratification approach conflicts"
            ]
        }

        return risk_factors.get(dimension, ["Cultural difference detected"])

    def _get_communication_challenges(self, differences: Dict) -> List[str]:
        """Identify communication challenges based on dimension differences."""
        challenges = []

        for dim_code, diff_data in differences.items():
            if diff_data["impact_level"] in ["significant", "major"]:
                if dim_code == "PDI":
                    challenges.append("Hierarchical communication style differences")
                elif dim_code == "IDV":
                    challenges.append("Individual vs collective communication preferences")
                elif dim_code == "UAI":
                    challenges.append("Different tolerance for ambiguous communication")
                elif dim_code == "LTO":
                    challenges.append("Short-term vs long-term communication focus")

        return challenges

    def _get_mitigation_strategies(self, high_risk_dimensions: List[Dict]) -> List[str]:
        """Generate mitigation strategies for cultural differences."""
        strategies = []

        for risk_dim in high_risk_dimensions:
            dim_code = risk_dim["dimension"]

            if dim_code == "PDI":
                strategies.append("Clarify authority and decision-making processes")
            elif dim_code == "IDV":
                strategies.append("Balance individual and group perspectives")
            elif dim_code == "MAS":
                strategies.append("Acknowledge different achievement orientations")
            elif dim_code == "UAI":
                strategies.append("Provide clear structure and reduce ambiguity")
            elif dim_code == "LTO":
                strategies.append("Acknowledge different time orientations")
            elif dim_code == "IVR":
                strategies.append("Respect different expression styles")

        # Add general strategies
        if len(high_risk_dimensions) > 2:
            strategies.append("Increase cultural awareness and sensitivity")
            strategies.append("Use neutral, explicit communication")

        return list(set(strategies))  # Remove duplicates

    def _calculate_risk_level(self, distance: float, high_risk_count: int) -> str:
        """Calculate overall communication risk level."""
        if distance <= 20 and high_risk_count <= 1:
            return "low"
        elif distance <= 40 and high_risk_count <= 2:
            return "medium"
        elif distance <= 60 and high_risk_count <= 4:
            return "high"
        else:
            return "very_high"

    def generate_radar_chart_data(
        self,
        cultures: List[Tuple[str, CulturalDimensions]]
    ) -> Dict[str, any]:
        """
        Generate data structure for cultural radar chart visualization.

        Args:
            cultures: List of (label, dimensions) tuples

        Returns:
            Radar chart data structure
        """
        chart_data = {
            "dimensions": list(self.DIMENSION_NAMES.keys()),
            "dimension_labels": list(self.DIMENSION_NAMES.values()),
            "cultures": []
        }

        for label, dimensions in cultures:
            culture_data = {
                "label": label,
                "values": dimensions.to_list(),
                "color": self._generate_culture_color(label)
            }
            chart_data["cultures"].append(culture_data)

        return chart_data

    def _generate_culture_color(self, label: str) -> str:
        """Generate a consistent color for a culture label."""
        color_map = {
            "DE": "#FF6B6B",
            "US": "#4ECDC4",
            "JP": "#45B7D1",
            "CN": "#96CEB4",
            "FR": "#FECA57",
            "IT": "#FF9FF3",
            "ES": "#54A0FF"
        }
        return color_map.get(label, "#95A5A6")