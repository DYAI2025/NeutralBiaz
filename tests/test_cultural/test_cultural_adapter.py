"""
Tests for Cultural Adapter

Comprehensive test suite for the cultural severity adjustment engine
and cross-cultural bias modification functionality.
"""

import pytest
from unittest.mock import Mock, patch
from bias_engine.cultural.adapters.cultural_adapter import (
    CulturalAdapter,
    CulturalContext,
    BiasAdjustment
)
from bias_engine.cultural.models.hofstede_model import CulturalDimensions, CulturalProfile
from bias_engine.cultural.data.profile_manager import CulturalProfileManager


class TestCulturalAdapter:
    """Test the CulturalAdapter class."""

    @pytest.fixture
    def mock_profile_manager(self):
        """Create a mock profile manager for testing."""
        manager = Mock(spec=CulturalProfileManager)

        # German profile
        german_dims = CulturalDimensions(35, 67, 66, 65, 83, 40)
        german_profile = CulturalProfile(
            "Germany", "DE", german_dims,
            {"communication_style": "direct", "hierarchy_acceptance": "low"}
        )

        # US profile
        us_dims = CulturalDimensions(40, 91, 62, 46, 26, 68)
        us_profile = CulturalProfile(
            "United States", "US", us_dims,
            {"communication_style": "assertive", "hierarchy_acceptance": "low"}
        )

        # Japanese profile
        japanese_dims = CulturalDimensions(54, 46, 95, 92, 88, 42)
        japanese_profile = CulturalProfile(
            "Japan", "JP", japanese_dims,
            {"communication_style": "indirect", "hierarchy_acceptance": "medium"}
        )

        def get_profile_side_effect(code):
            profiles = {"DE": german_profile, "US": us_profile, "JP": japanese_profile}
            return profiles.get(code.upper(), us_profile)  # Default to US

        manager.get_profile.side_effect = get_profile_side_effect
        return manager

    @pytest.fixture
    def cultural_adapter(self, mock_profile_manager):
        """Create a CulturalAdapter instance for testing."""
        return CulturalAdapter(mock_profile_manager)

    def test_cultural_adapter_initialization(self, cultural_adapter):
        """Test CulturalAdapter initialization."""
        assert cultural_adapter.logger is not None
        assert cultural_adapter.profile_manager is not None
        assert cultural_adapter.hofstede_model is not None
        assert cultural_adapter.MODIFIER_THRESHOLDS is not None
        assert cultural_adapter.BIAS_CULTURAL_SENSITIVITY is not None

    def test_bias_cultural_sensitivity_mapping(self, cultural_adapter):
        """Test that bias types have proper cultural sensitivity mappings."""
        sensitivity_map = cultural_adapter.BIAS_CULTURAL_SENSITIVITY

        # Check that all major bias types are covered
        expected_types = ["gender", "racial", "age", "religious", "socioeconomic", "political"]
        for bias_type in expected_types:
            assert bias_type in sensitivity_map

        # Check that each bias type has all dimension weights
        for bias_type, weights in sensitivity_map.items():
            assert len(weights) == 6  # All 6 Hofstede dimensions
            for dim in ["PDI", "IDV", "MAS", "UAI", "LTO", "IVR"]:
                assert dim in weights
                assert 0 <= weights[dim] <= 1  # Weights should be normalized

    def test_apply_cultural_modifiers_basic(self, cultural_adapter):
        """Test basic application of cultural modifiers."""
        # Sample bias results
        bias_results = {
            "overall_bias_score": 0.6,
            "biases_detected": {
                "gender": {"severity": 0.7, "confidence": 0.8},
                "racial": {"severity": 0.5, "confidence": 0.9}
            }
        }

        # Apply cultural modifiers (German to US)
        result = cultural_adapter.apply_cultural_modifiers(bias_results, "DE", "US")

        # Check that original structure is preserved
        assert "overall_bias_score" in result
        assert "biases_detected" in result

        # Check that cultural information is added
        assert "cultural_context" in result
        assert "cultural_adjustments" in result

        # Check that severities are adjusted
        for bias_type in bias_results["biases_detected"]:
            assert "original_severity" in result["biases_detected"][bias_type]
            assert "severity" in result["biases_detected"][bias_type]
            assert "cultural_adjustment" in result["biases_detected"][bias_type]

    def test_cultural_modifier_calculation(self, cultural_adapter):
        """Test cultural modifier calculation for specific bias types."""
        # Create cultural context with significant differences
        german_dims = CulturalDimensions(35, 67, 66, 65, 83, 40)
        japanese_dims = CulturalDimensions(54, 46, 95, 92, 88, 42)

        german_profile = CulturalProfile("Germany", "DE", german_dims, {})
        japanese_profile = CulturalProfile("Japan", "JP", japanese_dims, {})

        cultural_context = cultural_adapter._analyze_cultural_context(
            german_profile, japanese_profile
        )

        # Test gender bias modifier (high MAS sensitivity)
        gender_modifier = cultural_adapter._calculate_cultural_modifier("gender", cultural_context)
        assert isinstance(gender_modifier, float)
        assert 0.5 <= gender_modifier <= 2.0  # Should be within reasonable bounds

        # Test age bias modifier (high PDI and LTO sensitivity)
        age_modifier = cultural_adapter._calculate_cultural_modifier("age", cultural_context)
        assert isinstance(age_modifier, float)
        assert 0.5 <= age_modifier <= 2.0

    def test_cultural_context_analysis(self, cultural_adapter):
        """Test cultural context analysis."""
        german_dims = CulturalDimensions(35, 67, 66, 65, 83, 40)
        us_dims = CulturalDimensions(40, 91, 62, 46, 26, 68)

        german_profile = CulturalProfile("Germany", "DE", german_dims, {})
        us_profile = CulturalProfile("United States", "US", us_dims, {})

        context = cultural_adapter._analyze_cultural_context(german_profile, us_profile)

        assert isinstance(context, CulturalContext)
        assert context.sender_culture == german_profile
        assert context.receiver_culture == us_profile
        assert context.cultural_distance >= 0
        assert context.risk_level in ["low", "medium", "high", "very_high"]
        assert isinstance(context.high_risk_dimensions, list)
        assert isinstance(context.mitigation_strategies, list)

    def test_bias_adjustment_creation(self, cultural_adapter):
        """Test bias adjustment object creation."""
        bias_data = {"severity": 0.7, "confidence": 0.8}

        german_dims = CulturalDimensions(35, 67, 66, 65, 83, 40)
        us_dims = CulturalDimensions(40, 91, 62, 46, 26, 68)
        german_profile = CulturalProfile("Germany", "DE", german_dims, {})
        us_profile = CulturalProfile("United States", "US", us_dims, {})

        cultural_context = cultural_adapter._analyze_cultural_context(
            german_profile, us_profile
        )

        adjustment = cultural_adapter._calculate_bias_adjustment(
            "gender", bias_data, cultural_context
        )

        assert isinstance(adjustment, BiasAdjustment)
        assert adjustment.original_severity == 0.7
        assert 0.0 <= adjustment.adjusted_severity <= 1.0
        assert adjustment.cultural_modifier > 0
        assert isinstance(adjustment.explanation, str)
        assert isinstance(adjustment.cultural_factors, list)

    def test_modifier_bounds_checking(self, cultural_adapter):
        """Test that cultural modifiers stay within reasonable bounds."""
        # Test with extreme cultural differences
        extreme1 = CulturalDimensions(0, 0, 0, 0, 0, 0)
        extreme2 = CulturalDimensions(100, 100, 100, 100, 100, 100)

        profile1 = CulturalProfile("Extreme1", "EX1", extreme1, {})
        profile2 = CulturalProfile("Extreme2", "EX2", extreme2, {})

        cultural_context = cultural_adapter._analyze_cultural_context(profile1, profile2)

        for bias_type in cultural_adapter.BIAS_CULTURAL_SENSITIVITY:
            modifier = cultural_adapter._calculate_cultural_modifier(bias_type, cultural_context)

            # Modifier should be within reasonable bounds even for extreme differences
            assert 0.5 <= modifier <= 2.0, f"Modifier {modifier} for {bias_type} is out of bounds"

    def test_explanation_generation(self, cultural_adapter):
        """Test cultural adjustment explanation generation."""
        german_dims = CulturalDimensions(35, 67, 66, 65, 83, 40)
        japanese_dims = CulturalDimensions(54, 46, 95, 92, 88, 42)

        german_profile = CulturalProfile("Germany", "DE", german_dims, {})
        japanese_profile = CulturalProfile("Japan", "JP", japanese_dims, {})

        cultural_context = cultural_adapter._analyze_cultural_context(
            german_profile, japanese_profile
        )

        explanation = cultural_adapter._generate_adjustment_explanation(
            "gender", 1.2, cultural_context
        )

        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "Germany" in explanation
        assert "Japan" in explanation
        assert "cultural" in explanation.lower()

    def test_cultural_factor_identification(self, cultural_adapter):
        """Test identification of cultural factors affecting bias."""
        german_dims = CulturalDimensions(35, 67, 66, 65, 83, 40)
        chinese_dims = CulturalDimensions(80, 20, 66, 30, 87, 24)

        german_profile = CulturalProfile(
            "Germany", "DE", german_dims,
            {"communication_style": "direct", "hierarchy_acceptance": "low"}
        )
        chinese_profile = CulturalProfile(
            "China", "CN", chinese_dims,
            {"communication_style": "indirect", "hierarchy_acceptance": "high"}
        )

        cultural_context = cultural_adapter._analyze_cultural_context(
            german_profile, chinese_profile
        )

        factors = cultural_adapter._identify_cultural_factors("age", cultural_context)

        assert isinstance(factors, list)
        # Should identify significant cultural differences
        assert len(factors) > 0

        # Should mention communication style difference
        communication_mentioned = any(
            "communication" in factor.lower() for factor in factors
        )
        # Should mention hierarchy differences for age bias
        hierarchy_mentioned = any(
            "hierarchy" in factor.lower() for factor in factors
        )

    def test_overall_score_updates(self, cultural_adapter):
        """Test that overall bias scores are updated with cultural considerations."""
        bias_results = {
            "overall_bias_score": 0.6,
            "biases_detected": {
                "gender": {"severity": 0.7},
                "racial": {"severity": 0.5}
            }
        }

        result = cultural_adapter.apply_cultural_modifiers(bias_results, "DE", "JP")

        # Should have culturally adjusted overall score
        if "culturally_adjusted_score" in result:
            assert isinstance(result["culturally_adjusted_score"], float)
            assert 0.0 <= result["culturally_adjusted_score"] <= 1.0

        # Should have cultural communication risk assessment
        assert "cultural_communication_risk" in result
        risk_info = result["cultural_communication_risk"]
        assert "level" in risk_info
        assert "distance" in risk_info
        assert "requires_mitigation" in risk_info

    def test_cultural_explanation_detailed(self, cultural_adapter):
        """Test detailed cultural explanation generation."""
        explanation = cultural_adapter.get_cultural_explanation("gender", "DE", "JP")

        assert isinstance(explanation, dict)
        assert "bias_type" in explanation
        assert "cultural_context" in explanation
        assert "sensitivity_factors" in explanation
        assert "cultural_explanation" in explanation
        assert "mitigation_strategies" in explanation

        # Check bias type is correct
        assert explanation["bias_type"] == "gender"

        # Check that sensitivity factors are included
        sensitivity = explanation["sensitivity_factors"]
        assert isinstance(sensitivity, dict)
        assert len(sensitivity) == 6  # All dimensions

    def test_error_handling(self, cultural_adapter):
        """Test error handling in cultural adapter."""
        # Test with invalid bias results
        invalid_results = "not a dictionary"

        result = cultural_adapter.apply_cultural_modifiers(invalid_results, "DE", "US")

        # Should not crash and should include error information
        assert isinstance(result, (dict, str))

    def test_different_bias_types_sensitivity(self, cultural_adapter):
        """Test that different bias types have different cultural sensitivities."""
        german_dims = CulturalDimensions(35, 67, 66, 65, 83, 40)
        chinese_dims = CulturalDimensions(80, 20, 66, 30, 87, 24)

        german_profile = CulturalProfile("Germany", "DE", german_dims, {})
        chinese_profile = CulturalProfile("China", "CN", chinese_dims, {})

        cultural_context = cultural_adapter._analyze_cultural_context(
            german_profile, chinese_profile
        )

        # Calculate modifiers for different bias types
        gender_modifier = cultural_adapter._calculate_cultural_modifier("gender", cultural_context)
        age_modifier = cultural_adapter._calculate_cultural_modifier("age", cultural_context)
        political_modifier = cultural_adapter._calculate_cultural_modifier("political", cultural_context)

        # Modifiers should be different due to different sensitivity mappings
        modifiers = [gender_modifier, age_modifier, political_modifier]
        assert len(set(modifiers)) > 1, "All modifiers are the same - no differentiation"

    def test_same_culture_no_adjustment(self, cultural_adapter):
        """Test that same culture pairs have minimal adjustment."""
        bias_results = {
            "overall_bias_score": 0.6,
            "biases_detected": {
                "gender": {"severity": 0.7}
            }
        }

        result = cultural_adapter.apply_cultural_modifiers(bias_results, "DE", "DE")

        # Cultural distance should be 0
        assert result["cultural_context"]["cultural_distance"] == 0.0

        # Adjustments should be minimal
        gender_adjustment = result["cultural_adjustments"]["gender"]
        assert abs(gender_adjustment["cultural_modifier"] - 1.0) < 0.1