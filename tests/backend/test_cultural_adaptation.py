"""
Unit tests for cultural adaptation engine.
Tests cultural context processing, cross-cultural bias detection, and adaptation mechanisms.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'bias-engine', 'src'))

from bias_engine.cultural.adapter import CulturalAdapter
from bias_engine.cultural.models import CulturalProfile, CulturalContext
from bias_engine.core.models import BiasResult, AnalysisRequest
from bias_engine.cultural.exceptions import CulturalContextError


class TestCulturalAdapter:
    """Test suite for cultural adaptation functionality."""

    @pytest.fixture
    def adapter(self):
        """Create cultural adapter instance."""
        return CulturalAdapter()

    @pytest.fixture
    def cultural_profiles(self) -> Dict[str, CulturalProfile]:
        """Sample cultural profiles for testing."""
        return {
            "en-US": CulturalProfile(
                culture_code="en-US",
                individualism_score=0.91,
                power_distance=0.40,
                uncertainty_avoidance=0.46,
                masculine_feminine=0.62,
                bias_sensitivity={
                    "authority_bias": 0.3,
                    "conformity_bias": 0.2,
                    "group_think": 0.4
                },
                communication_style="direct",
                values=["independence", "achievement", "equality"]
            ),
            "ja-JP": CulturalProfile(
                culture_code="ja-JP",
                individualism_score=0.27,
                power_distance=0.54,
                uncertainty_avoidance=0.92,
                masculine_feminine=0.95,
                bias_sensitivity={
                    "authority_bias": 0.8,
                    "conformity_bias": 0.9,
                    "group_think": 0.7
                },
                communication_style="indirect",
                values=["harmony", "respect", "group_consensus"]
            ),
            "de-DE": CulturalProfile(
                culture_code="de-DE",
                individualism_score=0.67,
                power_distance=0.35,
                uncertainty_avoidance=0.65,
                masculine_feminine=0.66,
                bias_sensitivity={
                    "authority_bias": 0.4,
                    "conformity_bias": 0.3,
                    "group_think": 0.2
                },
                communication_style="direct",
                values=["efficiency", "order", "punctuality"]
            )
        }

    @pytest.fixture
    def cultural_test_cases(self) -> Dict[str, Dict[str, Any]]:
        """Test cases with cultural context variations."""
        return {
            "authority_deference": {
                "text": "The manager's decision is final and should not be questioned.",
                "expected_bias_scores": {
                    "en-US": 0.6,  # Moderate bias in individualistic culture
                    "ja-JP": 0.2,  # Low bias in hierarchical culture
                    "de-DE": 0.5   # Moderate bias
                }
            },
            "group_consensus": {
                "text": "We should all agree with the majority opinion to maintain harmony.",
                "expected_bias_scores": {
                    "en-US": 0.7,  # High bias in individualistic culture
                    "ja-JP": 0.3,  # Low bias in collectivistic culture
                    "de-DE": 0.6   # Moderate-high bias
                }
            },
            "individual_achievement": {
                "text": "Personal success is more important than group harmony.",
                "expected_bias_scores": {
                    "en-US": 0.2,  # Low bias in individualistic culture
                    "ja-JP": 0.8,  # High bias in collectivistic culture
                    "de-DE": 0.4   # Moderate bias
                }
            }
        }

    @pytest.mark.unit
    async def test_load_cultural_profiles(self, adapter, cultural_profiles):
        """Test loading and validation of cultural profiles."""
        # Mock the profile loading
        with patch.object(adapter, '_load_profiles') as mock_load:
            mock_load.return_value = cultural_profiles

            await adapter.initialize()

            assert adapter.is_initialized
            assert len(adapter.profiles) == 3
            assert "en-US" in adapter.profiles
            assert "ja-JP" in adapter.profiles
            assert "de-DE" in adapter.profiles

    @pytest.mark.unit
    async def test_cultural_context_adaptation(self, adapter, cultural_profiles, cultural_test_cases):
        """Test adaptation of bias detection based on cultural context."""
        adapter.profiles = cultural_profiles
        adapter.is_initialized = True

        test_case = cultural_test_cases["authority_deference"]

        for culture, expected_score in test_case["expected_bias_scores"].items():
            context = CulturalContext(
                culture_code=culture,
                profile=cultural_profiles[culture]
            )

            # Test cultural adaptation
            adapted_result = await adapter.adapt_bias_detection(
                text=test_case["text"],
                cultural_context=context,
                base_bias_score=0.5
            )

            # Verify adaptation affects the bias score appropriately
            assert abs(adapted_result.cultural_adjusted_score - expected_score) < 0.3
            assert adapted_result.cultural_factors is not None
            assert culture in adapted_result.cultural_factors

    @pytest.mark.unit
    async def test_individualism_collectivism_adaptation(self, adapter, cultural_profiles):
        """Test adaptation based on individualism vs collectivism."""
        adapter.profiles = cultural_profiles
        adapter.is_initialized = True

        # Test text that varies by individualism
        group_text = "The team's decision is more important than individual opinions."

        # US context (high individualism)
        us_context = CulturalContext(
            culture_code="en-US",
            profile=cultural_profiles["en-US"]
        )
        us_result = await adapter.adapt_bias_detection(group_text, us_context, 0.5)

        # Japanese context (low individualism)
        jp_context = CulturalContext(
            culture_code="ja-JP",
            profile=cultural_profiles["ja-JP"]
        )
        jp_result = await adapter.adapt_bias_detection(group_text, jp_context, 0.5)

        # Should have higher bias score in individualistic culture
        assert us_result.cultural_adjusted_score > jp_result.cultural_adjusted_score

    @pytest.mark.unit
    async def test_power_distance_adaptation(self, adapter, cultural_profiles):
        """Test adaptation based on power distance cultural dimension."""
        adapter.profiles = cultural_profiles
        adapter.is_initialized = True

        # Test text about hierarchy
        hierarchy_text = "Subordinates should always follow orders without question."

        # Low power distance (US/Germany)
        us_context = CulturalContext("en-US", cultural_profiles["en-US"])
        us_result = await adapter.adapt_bias_detection(hierarchy_text, us_context, 0.5)

        # Higher power distance (Japan)
        jp_context = CulturalContext("ja-JP", cultural_profiles["ja-JP"])
        jp_result = await adapter.adapt_bias_detection(hierarchy_text, jp_context, 0.5)

        # Should show more bias concern in low power distance culture
        assert us_result.cultural_adjusted_score > jp_result.cultural_adjusted_score

    @pytest.mark.unit
    async def test_uncertainty_avoidance_adaptation(self, adapter, cultural_profiles):
        """Test adaptation based on uncertainty avoidance."""
        adapter.profiles = cultural_profiles
        adapter.is_initialized = True

        # Test text about risk and uncertainty
        risk_text = "We should stick with traditional methods rather than try new approaches."

        # Low uncertainty avoidance (US)
        us_context = CulturalContext("en-US", cultural_profiles["en-US"])
        us_result = await adapter.adapt_bias_detection(risk_text, us_context, 0.5)

        # High uncertainty avoidance (Japan)
        jp_context = CulturalContext("ja-JP", cultural_profiles["ja-JP"])
        jp_result = await adapter.adapt_bias_detection(risk_text, jp_context, 0.5)

        # Should be less biased in high uncertainty avoidance culture
        assert us_result.cultural_adjusted_score > jp_result.cultural_adjusted_score

    @pytest.mark.unit
    async def test_communication_style_adaptation(self, adapter, cultural_profiles):
        """Test adaptation based on direct vs indirect communication styles."""
        adapter.profiles = cultural_profiles
        adapter.is_initialized = True

        # Direct confrontational text
        direct_text = "Your proposal is wrong and should be rejected immediately."

        # Indirect suggestion text
        indirect_text = "Perhaps we might consider alternative approaches that could be more suitable."

        # Direct culture context (US/Germany)
        us_context = CulturalContext("en-US", cultural_profiles["en-US"])
        direct_in_us = await adapter.adapt_bias_detection(direct_text, us_context, 0.5)
        indirect_in_us = await adapter.adapt_bias_detection(indirect_text, us_context, 0.5)

        # Indirect culture context (Japan)
        jp_context = CulturalContext("ja-JP", cultural_profiles["ja-JP"])
        direct_in_jp = await adapter.adapt_bias_detection(direct_text, jp_context, 0.5)
        indirect_in_jp = await adapter.adapt_bias_detection(indirect_text, jp_context, 0.5)

        # Direct text should be more problematic in indirect culture
        assert direct_in_jp.cultural_adjusted_score > direct_in_us.cultural_adjusted_score

    @pytest.mark.unit
    async def test_cultural_values_alignment(self, adapter, cultural_profiles):
        """Test bias detection alignment with cultural values."""
        adapter.profiles = cultural_profiles
        adapter.is_initialized = True

        values_texts = {
            "independence": "Individual achievement is the highest priority.",
            "harmony": "Group consensus is more important than individual opinions.",
            "efficiency": "Processes should be optimized for maximum productivity."
        }

        # Test each text against each culture
        for value, text in values_texts.items():
            results = {}
            for culture_code, profile in cultural_profiles.items():
                context = CulturalContext(culture_code, profile)
                result = await adapter.adapt_bias_detection(text, context, 0.5)
                results[culture_code] = result

            # Verify that cultures with aligned values show lower bias scores
            if value == "independence":
                assert results["en-US"].cultural_adjusted_score < results["ja-JP"].cultural_adjusted_score
            elif value == "harmony":
                assert results["ja-JP"].cultural_adjusted_score < results["en-US"].cultural_adjusted_score
            elif value == "efficiency":
                assert results["de-DE"].cultural_adjusted_score <= max(results["en-US"].cultural_adjusted_score,
                                                                      results["ja-JP"].cultural_adjusted_score)

    @pytest.mark.unit
    async def test_fallback_to_default_culture(self, adapter, cultural_profiles):
        """Test fallback behavior for unknown cultural contexts."""
        adapter.profiles = cultural_profiles
        adapter.is_initialized = True

        # Test with unknown culture code
        unknown_context = CulturalContext(
            culture_code="xx-XX",
            profile=None
        )

        text = "This is a test text for cultural adaptation."

        # Should fallback to default (en-US) without error
        result = await adapter.adapt_bias_detection(text, unknown_context, 0.5)
        assert result is not None
        assert result.cultural_factors is not None
        assert "fallback_used" in result.metadata

    @pytest.mark.unit
    async def test_cultural_bias_sensitivity_mapping(self, adapter, cultural_profiles):
        """Test mapping of bias types to cultural sensitivity levels."""
        adapter.profiles = cultural_profiles
        adapter.is_initialized = True

        # Test authority bias sensitivity
        authority_text = "The boss is always right and should not be questioned."

        us_context = CulturalContext("en-US", cultural_profiles["en-US"])
        jp_context = CulturalContext("ja-JP", cultural_profiles["ja-JP"])

        us_result = await adapter.adapt_bias_detection(authority_text, us_context, 0.6)
        jp_result = await adapter.adapt_bias_detection(authority_text, jp_context, 0.6)

        # Japan has higher authority bias sensitivity, so should show lower adjusted score
        assert us_result.cultural_adjusted_score > jp_result.cultural_adjusted_score

        # Check that cultural factors explain the difference
        assert "authority_bias_sensitivity" in us_result.cultural_factors
        assert "authority_bias_sensitivity" in jp_result.cultural_factors

    @pytest.mark.unit
    async def test_cross_cultural_comparison(self, adapter, cultural_profiles):
        """Test cross-cultural comparison functionality."""
        adapter.profiles = cultural_profiles
        adapter.is_initialized = True

        text = "Individual performance should always be prioritized over team harmony."

        # Get results for multiple cultures
        comparison_results = await adapter.cross_cultural_analysis(
            text=text,
            cultures=["en-US", "ja-JP", "de-DE"],
            base_bias_score=0.6
        )

        assert len(comparison_results) == 3
        assert "en-US" in comparison_results
        assert "ja-JP" in comparison_results
        assert "de-DE" in comparison_results

        # Verify cultural differences
        us_score = comparison_results["en-US"].cultural_adjusted_score
        jp_score = comparison_results["ja-JP"].cultural_adjusted_score

        # Individual performance focus should be less biased in US than Japan
        assert us_score < jp_score

        # Check cultural explanations
        for culture, result in comparison_results.items():
            assert result.cultural_explanation is not None
            assert len(result.cultural_explanation) > 0

    @pytest.mark.unit
    async def test_cultural_context_validation(self, adapter):
        """Test validation of cultural context parameters."""
        # Test invalid culture codes
        invalid_contexts = [
            None,
            "",
            "invalid",
            "xx",
            "en-INVALID",
            "123-456"
        ]

        for invalid_context in invalid_contexts:
            with pytest.raises((CulturalContextError, ValueError)):
                context = CulturalContext(culture_code=invalid_context, profile=None)
                await adapter.validate_cultural_context(context)

    @pytest.mark.unit
    async def test_cultural_profile_completeness(self, adapter, cultural_profiles):
        """Test validation of cultural profile completeness."""
        adapter.profiles = cultural_profiles

        for culture_code, profile in cultural_profiles.items():
            # Validate required fields
            assert profile.culture_code == culture_code
            assert 0.0 <= profile.individualism_score <= 1.0
            assert 0.0 <= profile.power_distance <= 1.0
            assert 0.0 <= profile.uncertainty_avoidance <= 1.0
            assert 0.0 <= profile.masculine_feminine <= 1.0

            # Validate bias sensitivity mappings
            assert isinstance(profile.bias_sensitivity, dict)
            assert len(profile.bias_sensitivity) > 0

            for bias_type, sensitivity in profile.bias_sensitivity.items():
                assert 0.0 <= sensitivity <= 1.0

            # Validate cultural values
            assert isinstance(profile.values, list)
            assert len(profile.values) > 0

    @pytest.mark.integration
    async def test_cultural_adaptation_integration(self, adapter, cultural_profiles):
        """Test integration with full bias detection workflow."""
        adapter.profiles = cultural_profiles
        adapter.is_initialized = True

        # Test complete workflow
        test_scenarios = [
            {
                "text": "The senior management's strategic direction should be followed without deviation.",
                "cultures": ["en-US", "ja-JP", "de-DE"],
                "expected_pattern": "ja-JP < de-DE < en-US"  # Authority acceptance order
            },
            {
                "text": "Individual creativity should override established procedures.",
                "cultures": ["en-US", "ja-JP", "de-DE"],
                "expected_pattern": "en-US < de-DE < ja-JP"  # Individual vs procedure preference
            }
        ]

        for scenario in test_scenarios:
            results = {}

            for culture in scenario["cultures"]:
                context = CulturalContext(culture, cultural_profiles[culture])
                result = await adapter.adapt_bias_detection(
                    text=scenario["text"],
                    cultural_context=context,
                    base_bias_score=0.5
                )
                results[culture] = result.cultural_adjusted_score

            # Verify expected cultural pattern
            if scenario["expected_pattern"] == "ja-JP < de-DE < en-US":
                assert results["ja-JP"] < results["de-DE"] < results["en-US"]
            elif scenario["expected_pattern"] == "en-US < de-DE < ja-JP":
                assert results["en-US"] < results["de-DE"] < results["ja-JP"]

    @pytest.mark.unit
    async def test_cultural_explanation_generation(self, adapter, cultural_profiles):
        """Test generation of cultural explanations for bias adjustments."""
        adapter.profiles = cultural_profiles
        adapter.is_initialized = True

        text = "The team leader's authority should never be questioned publicly."

        for culture_code, profile in cultural_profiles.items():
            context = CulturalContext(culture_code, profile)
            result = await adapter.adapt_bias_detection(text, context, 0.6)

            # Should have cultural explanation
            assert result.cultural_explanation is not None
            assert len(result.cultural_explanation) > 50  # Substantial explanation

            # Should mention relevant cultural factors
            explanation = result.cultural_explanation.lower()

            if culture_code == "ja-JP":
                assert any(word in explanation for word in ["hierarchy", "authority", "respect", "power"])
            elif culture_code == "en-US":
                assert any(word in explanation for word in ["individual", "question", "challenge", "equality"])
            elif culture_code == "de-DE":
                assert any(word in explanation for word in ["order", "structure", "efficiency"])


# Performance tests for cultural adaptation
@pytest.mark.slow
class TestCulturalAdaptationPerformance:
    """Performance tests for cultural adaptation system."""

    @pytest.mark.asyncio
    async def test_cultural_adaptation_performance(self):
        """Test performance of cultural adaptation with multiple cultures."""
        adapter = CulturalAdapter()
        await adapter.initialize()

        # Test with multiple cultures simultaneously
        cultures = ["en-US", "ja-JP", "de-DE", "fr-FR", "zh-CN"]
        test_texts = [
            "Authority figures should be respected without question.",
            "Individual achievement is more important than group harmony.",
            "Established procedures should always be followed.",
            "Innovation requires challenging traditional methods.",
        ] * 5  # 20 texts total

        import time
        start_time = time.time()

        # Process all combinations
        tasks = []
        for text in test_texts:
            for culture in cultures:
                try:
                    context = CulturalContext(culture, adapter.profiles.get(culture))
                    task = adapter.adapt_bias_detection(text, context, 0.5)
                    tasks.append(task)
                except Exception:
                    continue  # Skip unavailable cultures

        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_time = end_time - start_time

        successful_results = [r for r in results if not isinstance(r, Exception)]
        throughput = len(successful_results) / total_time

        print(f"Processed {len(successful_results)} cultural adaptations in {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} adaptations/second")

        # Performance targets
        assert throughput >= 10.0  # Should process at least 10 adaptations per second
        assert len(successful_results) >= len(tasks) * 0.8  # At least 80% success rate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])