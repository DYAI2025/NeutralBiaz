"""
Tests for Cultural Integration System

Comprehensive test suite for the cultural integration module
and its seamless integration with bias detection systems.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from bias_engine.cultural.integration import (
    CulturalIntegration,
    CulturalIntegrationError,
    with_cultural_context,
    cultural_bias_enhancer,
    enhance_bias_with_culture,
    get_cultural_context,
    get_cultural_integration,
    initialize_cultural_integration
)


class TestCulturalIntegration:
    """Test the CulturalIntegration class."""

    @pytest.fixture
    def cultural_integration(self):
        """Create a CulturalIntegration instance for testing."""
        with patch('bias_engine.cultural.integration.CulturalProfileManager'):
            return CulturalIntegration(enable_caching=True)

    def test_cultural_integration_initialization(self, cultural_integration):
        """Test CulturalIntegration initialization."""
        assert cultural_integration.profile_manager is not None
        assert cultural_integration.hofstede_model is not None
        assert cultural_integration.cultural_adapter is not None
        assert cultural_integration.cultural_analyzer is not None
        assert cultural_integration.cultural_intelligence is not None
        assert cultural_integration.enable_caching is True
        assert cultural_integration._cache == {}

    def test_hooks_registration(self, cultural_integration):
        """Test hook registration functionality."""
        # Test registering valid hooks
        def dummy_pre_hook(*args, **kwargs):
            pass

        def dummy_post_hook(*args, **kwargs):
            pass

        cultural_integration.register_pre_bias_hook(dummy_pre_hook)
        cultural_integration.register_post_bias_hook(dummy_post_hook)

        assert len(cultural_integration._pre_bias_hooks) == 1
        assert len(cultural_integration._post_bias_hooks) == 1

        # Test registering invalid hooks
        with pytest.raises(ValueError):
            cultural_integration.register_pre_bias_hook("not a function")

        with pytest.raises(ValueError):
            cultural_integration.register_post_bias_hook(123)

    @patch('bias_engine.cultural.integration.CulturalAdapter')
    @patch('bias_engine.cultural.integration.CulturalAnalyzer')
    @patch('bias_engine.cultural.integration.CulturalIntelligence')
    def test_enhance_bias_detection(self, mock_intelligence, mock_analyzer, mock_adapter, cultural_integration):
        """Test the main bias enhancement functionality."""
        # Setup mocks
        mock_adapter_instance = mock_adapter.return_value
        mock_analyzer_instance = mock_analyzer.return_value
        mock_intelligence_instance = mock_intelligence.return_value

        # Mock adapter response
        mock_adapter_instance.apply_cultural_modifiers.return_value = {
            "overall_bias_score": 0.7,
            "biases_detected": {"gender": {"severity": 0.8}},
            "cultural_context": {"distance": 25.5}
        }

        # Mock analyzer response
        mock_analysis = Mock()
        mock_analysis.overall_risk.value = "medium"
        mock_analysis.cultural_distance = 25.5
        mock_analysis.bridge_score = 74.5
        mock_analysis.insights = []
        mock_analysis.recommendations = ["Test recommendation"]
        mock_analysis.potential_misunderstandings = ["Test misunderstanding"]
        mock_analyzer_instance.analyze_cross_cultural_context.return_value = mock_analysis

        # Mock intelligence responses
        mock_intelligence_instance.generate_sensitivity_warnings.return_value = [
            {"type": "test_warning", "severity": "medium"}
        ]
        mock_intelligence_instance.recommend_communication_strategies.return_value = {
            "sender_adaptations": ["Test adaptation"]
        }

        # Test data
        bias_results = {
            "overall_bias_score": 0.6,
            "biases_detected": {"gender": {"severity": 0.7}}
        }

        # Execute enhancement
        result = cultural_integration.enhance_bias_detection(
            bias_results, "DE", "US", {"type": "business"}
        )

        # Verify structure
        assert "overall_bias_score" in result
        assert "biases_detected" in result
        assert "cultural_context" in result
        assert "cross_cultural_analysis" in result
        assert "cultural_intelligence" in result

        # Verify cross-cultural analysis
        analysis = result["cross_cultural_analysis"]
        assert analysis["overall_risk"] == "medium"
        assert analysis["cultural_distance"] == 25.5
        assert analysis["bridge_score"] == 74.5

        # Verify cultural intelligence
        intelligence = result["cultural_intelligence"]
        assert "sensitivity_warnings" in intelligence
        assert "communication_strategies" in intelligence

    def test_caching_functionality(self, cultural_integration):
        """Test caching behavior."""
        with patch.object(cultural_integration.cultural_adapter, 'apply_cultural_modifiers') as mock_adapter:
            with patch.object(cultural_integration.cultural_analyzer, 'analyze_cross_cultural_context') as mock_analyzer:
                with patch.object(cultural_integration.cultural_intelligence, 'generate_sensitivity_warnings') as mock_intelligence:
                    with patch.object(cultural_integration.cultural_intelligence, 'recommend_communication_strategies') as mock_strategies:

                        # Setup mocks
                        mock_adapter.return_value = {"test": "data"}
                        mock_analysis = Mock()
                        mock_analysis.overall_risk.value = "low"
                        mock_analysis.cultural_distance = 10.0
                        mock_analysis.bridge_score = 90.0
                        mock_analysis.insights = []
                        mock_analysis.recommendations = []
                        mock_analysis.potential_misunderstandings = []
                        mock_analyzer.return_value = mock_analysis
                        mock_intelligence.return_value = []
                        mock_strategies.return_value = {}

                        bias_results = {"overall_bias_score": 0.5}

                        # First call - should hit the methods
                        result1 = cultural_integration.enhance_bias_detection(bias_results, "DE", "US")

                        # Second call with same parameters - should use cache
                        result2 = cultural_integration.enhance_bias_detection(bias_results, "DE", "US")

                        # Verify cache is working (methods called only once)
                        assert mock_adapter.call_count == 1
                        assert result1 == result2

                        # Clear cache and verify
                        cultural_integration.clear_cache()
                        assert len(cultural_integration._cache) == 0

    def test_analyze_cultural_context_without_bias(self, cultural_integration):
        """Test cultural context analysis without bias results."""
        with patch.object(cultural_integration.cultural_analyzer, 'analyze_cross_cultural_context') as mock_analyzer:
            with patch.object(cultural_integration.cultural_intelligence, 'recommend_communication_strategies') as mock_strategies:
                with patch.object(cultural_integration.cultural_intelligence, 'generate_sensitivity_warnings') as mock_warnings:

                    # Setup mocks
                    mock_analysis = Mock()
                    mock_analysis.cultural_distance = 30.0
                    mock_analysis.bridge_score = 70.0
                    mock_analysis.overall_risk.value = "medium"
                    mock_analysis.insights = []
                    mock_analysis.recommendations = ["Test rec"]
                    mock_analysis.potential_misunderstandings = ["Test misunderstanding"]
                    mock_analyzer.return_value = mock_analysis

                    mock_strategies.return_value = {"test": "strategies"}
                    mock_warnings.return_value = [{"type": "test", "severity": "low"}]

                    result = cultural_integration.analyze_cultural_context(
                        "DE", "US", {"type": "educational"}
                    )

                    # Verify structure
                    assert "cultural_distance" in result
                    assert "bridge_score" in result
                    assert "overall_risk" in result
                    assert "insights" in result
                    assert "recommendations" in result
                    assert "communication_strategies" in result
                    assert "sensitivity_warnings" in result

                    # Verify values
                    assert result["cultural_distance"] == 30.0
                    assert result["bridge_score"] == 70.0
                    assert result["overall_risk"] == "medium"

    def test_error_handling(self, cultural_integration):
        """Test error handling in integration methods."""
        # Test with None profile manager causing errors
        with patch.object(cultural_integration.cultural_adapter, 'apply_cultural_modifiers',
                          side_effect=Exception("Test error")):

            with pytest.raises(CulturalIntegrationError):
                cultural_integration.enhance_bias_detection({}, "DE", "US")

        # Test with invalid cultural context analysis
        with patch.object(cultural_integration.cultural_analyzer, 'analyze_cross_cultural_context',
                          side_effect=Exception("Analysis error")):

            with pytest.raises(CulturalIntegrationError):
                cultural_integration.analyze_cultural_context("DE", "US")

    def test_validation_methods(self, cultural_integration):
        """Test culture code validation methods."""
        with patch.object(cultural_integration.profile_manager, 'get_supported_cultures',
                          return_value=['DE', 'US', 'JP', 'CN']):

            # Test valid codes
            assert cultural_integration.validate_culture_codes("DE", "US") is True
            assert cultural_integration.validate_culture_codes("JP") is True

            # Test invalid codes
            assert cultural_integration.validate_culture_codes("XX", "YY") is False
            assert cultural_integration.validate_culture_codes("DE", "XX") is False

            # Test get supported cultures
            supported = cultural_integration.get_supported_cultures()
            assert "DE" in supported
            assert "US" in supported

    def test_statistics_method(self, cultural_integration):
        """Test cultural statistics generation."""
        with patch.object(cultural_integration.profile_manager, 'get_supported_cultures',
                          return_value=['DE', 'US', 'JP']):
            with patch.object(cultural_integration.profile_manager, 'get_statistics',
                              return_value={"test": "stats"}):

                stats = cultural_integration.get_cultural_statistics()

                assert "supported_cultures" in stats
                assert "profile_manager_stats" in stats
                assert "cache_size" in stats
                assert "hooks" in stats

                # Check values
                assert stats["supported_cultures"] == 3
                assert stats["cache_size"] == 0  # Empty cache
                assert "pre_bias" in stats["hooks"]
                assert "post_bias" in stats["hooks"]


class TestCulturalDecorators:
    """Test cultural integration decorators."""

    def test_with_cultural_context_decorator(self):
        """Test the with_cultural_context decorator."""
        @with_cultural_context()
        def sample_function(text, sender_culture=None, receiver_culture=None, context=None):
            return {"original": "result", "text": text}

        with patch('bias_engine.cultural.integration.CulturalIntegration') as mock_integration:
            mock_instance = mock_integration.return_value
            mock_instance.analyze_cultural_context.return_value = {"test": "context"}

            # Test with cultural parameters
            result = sample_function("test", sender_culture="DE", receiver_culture="US")

            assert "original" in result
            assert "cultural_context" in result
            mock_instance.analyze_cultural_context.assert_called_once_with("DE", "US", None)

            # Test without cultural parameters
            result2 = sample_function("test")
            assert "cultural_context" not in result2

    def test_cultural_bias_enhancer_decorator(self):
        """Test the cultural_bias_enhancer decorator."""
        def mock_bias_detection(*args, **kwargs):
            return {"bias_score": 0.7, "biases": {"gender": 0.8}}

        enhanced_function = cultural_bias_enhancer(mock_bias_detection)

        with patch('bias_engine.cultural.integration.CulturalIntegration') as mock_integration:
            mock_instance = mock_integration.return_value
            mock_instance.enhance_bias_detection.return_value = {"enhanced": True}

            # Test with cultural parameters
            result = enhanced_function(sender_culture="DE", receiver_culture="US")

            assert result == {"enhanced": True}
            mock_instance.enhance_bias_detection.assert_called_once()

            # Test without cultural parameters
            result2 = enhanced_function()
            assert "bias_score" in result2  # Original result


class TestConvenienceFunctions:
    """Test convenience functions for cultural integration."""

    @patch('bias_engine.cultural.integration.get_cultural_integration')
    def test_enhance_bias_with_culture(self, mock_get_integration):
        """Test enhance_bias_with_culture convenience function."""
        mock_integration = Mock()
        mock_integration.enhance_bias_detection.return_value = {"enhanced": True}
        mock_get_integration.return_value = mock_integration

        bias_results = {"bias_score": 0.6}
        result = enhance_bias_with_culture(bias_results, "DE", "US")

        assert result == {"enhanced": True}
        mock_integration.enhance_bias_detection.assert_called_once_with(
            bias_results, "DE", "US", None
        )

    @patch('bias_engine.cultural.integration.get_cultural_integration')
    def test_get_cultural_context(self, mock_get_integration):
        """Test get_cultural_context convenience function."""
        mock_integration = Mock()
        mock_integration.analyze_cultural_context.return_value = {"context": True}
        mock_get_integration.return_value = mock_integration

        result = get_cultural_context("DE", "US", {"type": "business"})

        assert result == {"context": True}
        mock_integration.analyze_cultural_context.assert_called_once_with(
            "DE", "US", {"type": "business"}
        )


class TestGlobalInstance:
    """Test global instance management."""

    def test_global_instance_creation(self):
        """Test global instance creation and access."""
        # Clear any existing global instance
        import bias_engine.cultural.integration as integration_module
        integration_module._global_cultural_integration = None

        # First call should create instance
        instance1 = get_cultural_integration()
        assert instance1 is not None

        # Second call should return same instance
        instance2 = get_cultural_integration()
        assert instance1 is instance2

    def test_global_instance_initialization(self):
        """Test global instance initialization with parameters."""
        import bias_engine.cultural.integration as integration_module

        # Initialize with custom parameters
        instance = initialize_cultural_integration(enable_caching=False)

        assert instance.enable_caching is False

        # Get instance should return the initialized one
        same_instance = get_cultural_integration()
        assert instance is same_instance


@pytest.mark.integration
class TestCulturalIntegrationIntegration:
    """Integration tests for cultural integration system."""

    def test_end_to_end_bias_enhancement(self):
        """Test end-to-end bias enhancement workflow."""
        # This test would require actual profile data and models
        # For now, we'll test the structure and flow

        bias_results = {
            "overall_bias_score": 0.6,
            "biases_detected": {
                "gender": {"severity": 0.7, "confidence": 0.9},
                "racial": {"severity": 0.5, "confidence": 0.8}
            },
            "analysis_metadata": {"model": "test", "version": "1.0"}
        }

        with patch('bias_engine.cultural.integration.CulturalProfileManager'):
            integration = CulturalIntegration(enable_caching=True)

            with patch.object(integration, 'validate_culture_codes', return_value=True):
                with patch.object(integration.cultural_adapter, 'apply_cultural_modifiers',
                                  return_value=bias_results):
                    with patch.object(integration.cultural_analyzer, 'analyze_cross_cultural_context') as mock_analyzer:
                        with patch.object(integration.cultural_intelligence, 'generate_sensitivity_warnings',
                                          return_value=[]):
                            with patch.object(integration.cultural_intelligence, 'recommend_communication_strategies',
                                              return_value={}):

                                # Setup analyzer mock
                                mock_analysis = Mock()
                                mock_analysis.overall_risk.value = "medium"
                                mock_analysis.cultural_distance = 30.0
                                mock_analysis.bridge_score = 70.0
                                mock_analysis.insights = []
                                mock_analysis.recommendations = []
                                mock_analysis.potential_misunderstandings = []
                                mock_analyzer.return_value = mock_analysis

                                result = integration.enhance_bias_detection(
                                    bias_results, "DE", "US", {"type": "business"}
                                )

                                # Verify the enhancement workflow completed
                                assert isinstance(result, dict)
                                assert "cross_cultural_analysis" in result
                                assert "cultural_intelligence" in result

    def test_hook_execution_flow(self):
        """Test that hooks are executed in the correct order."""
        execution_order = []

        def pre_hook(*args, **kwargs):
            execution_order.append("pre")

        def post_hook(*args, **kwargs):
            execution_order.append("post")

        with patch('bias_engine.cultural.integration.CulturalProfileManager'):
            integration = CulturalIntegration()

            integration.register_pre_bias_hook(pre_hook)
            integration.register_post_bias_hook(post_hook)

            with patch.object(integration.cultural_adapter, 'apply_cultural_modifiers',
                              return_value={}):
                with patch.object(integration.cultural_analyzer, 'analyze_cross_cultural_context') as mock_analyzer:
                    with patch.object(integration.cultural_intelligence, 'generate_sensitivity_warnings',
                                      return_value=[]):
                        with patch.object(integration.cultural_intelligence, 'recommend_communication_strategies',
                                          return_value={}):

                            # Setup analyzer mock
                            mock_analysis = Mock()
                            mock_analysis.overall_risk.value = "low"
                            mock_analysis.cultural_distance = 10.0
                            mock_analysis.bridge_score = 90.0
                            mock_analysis.insights = []
                            mock_analysis.recommendations = []
                            mock_analysis.potential_misunderstandings = []
                            mock_analyzer.return_value = mock_analysis

                            integration.enhance_bias_detection({}, "DE", "US")

                            # Verify execution order
                            assert execution_order == ["pre", "post"]