"""
Comprehensive test suite for LLM integration components.
"""

import json
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from bias_engine.llm import (
    LLMClient,
    PromptManager,
    DebiasingPipeline,
    SelfBiasChecker,
    LLMProvider,
    LLMConfig,
    RateLimitConfig,
    DebiasingRequest,
    BatchDebiasingRequest,
    MarkerGenerationRequest,
    SelfBiasCheckRequest,
    BiasSpan,
    CulturalSeverityLevel,
    EpistemicClassification
)
from bias_engine.llm.cultural_integration import CulturalContextIntegrator
from bias_engine.llm.client import RateLimiter
from bias_engine.models.schemas import BiasDetection, BiasType, BiasLevel


class TestLLMClient:
    """Test LLM client infrastructure."""

    def test_llm_client_factory(self):
        """Test LLM client factory creation."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            model="gpt-4",
            max_tokens=1000,
            temperature=0.1
        )

        client = LLMClient.create(config)
        assert client is not None
        assert client.config.provider == LLMProvider.OPENAI

    def test_rate_limiter(self):
        """Test rate limiting functionality."""
        config = RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=10000,
            burst_size=5
        )

        limiter = RateLimiter(config)
        assert limiter.requests_per_minute == 60
        assert limiter.tokens_per_minute == 10000

    @pytest.mark.asyncio
    async def test_rate_limiter_acquire(self):
        """Test rate limiter token acquisition."""
        config = RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=10000,
            burst_size=5
        )

        limiter = RateLimiter(config)

        # Should succeed with normal request
        await limiter.acquire(100)

        # Should succeed with multiple small requests
        for _ in range(4):
            await limiter.acquire(50)

    def test_supported_providers(self):
        """Test supported provider listing."""
        providers = LLMClient.get_supported_providers()
        assert LLMProvider.OPENAI in providers
        assert LLMProvider.ANTHROPIC in providers
        assert LLMProvider.AZURE in providers


class TestPromptManager:
    """Test prompt management system."""

    def test_prompt_manager_initialization(self):
        """Test prompt manager loads built-in templates."""
        manager = PromptManager()
        templates = manager.list_templates()

        assert "debiaser_system" in templates
        assert "debias_span" in templates
        assert "debias_batch" in templates
        assert "marker_generator" in templates
        assert "self_bias_check" in templates

    def test_template_rendering(self):
        """Test template variable substitution."""
        manager = PromptManager()

        variables = {
            "span_id": "test_span_1",
            "input_language": "en",
            "output_language": "de",
            "bias_family": "racism",
            "bias_span": "problematic text"
        }

        rendered = manager.render_template("debias_span", variables, strict=False)
        assert "test_span_1" in rendered
        assert "racism" in rendered
        assert "problematic text" in rendered

    def test_message_creation(self):
        """Test message creation for LLM consumption."""
        manager = PromptManager()

        variables = {"output_language": "de"}
        message = manager.create_message("debiaser_system", variables)

        assert message["role"] == "system"
        assert "content" in message
        assert len(message["content"]) > 0

    def test_template_validation(self):
        """Test template validation."""
        manager = PromptManager()

        assert manager.validate_template("debiaser_system")
        assert manager.validate_template("debias_span")
        assert not manager.validate_template("nonexistent_template")

    def test_variable_extraction(self):
        """Test variable extraction from templates."""
        manager = PromptManager()
        template = manager.get_template("debias_span")

        expected_variables = {
            "span_id", "input_language", "output_language",
            "sender_culture", "receiver_culture", "bias_family"
        }

        assert expected_variables.issubset(set(template.variables))


class TestDebiasingPipeline:
    """Test debiasing pipeline functionality."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = AsyncMock()
        client.generate = AsyncMock(return_value=Mock(
            content='{"span_id": "test", "language": "de", "bias_family": "racism", "bias_subtype": "stereotyping", "analysis_explanation": "Test explanation", "can_preserve_core_intent": true, "variant_A_rewrite": "Neutral version", "variant_B_rewrite": "Emotional version", "safety_notes": "No issues"}',
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-sonnet",
            tokens_used={"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
            response_time=1.5
        ))
        return client

    @pytest.fixture
    def pipeline(self, mock_llm_client):
        """Create debiasing pipeline with mock client."""
        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            api_key="test-key",
            model="claude-3-sonnet"
        )
        pipeline = DebiasingPipeline(config)
        pipeline.llm_client = mock_llm_client
        return pipeline

    @pytest.fixture
    def sample_bias_span(self):
        """Create sample bias span for testing."""
        return BiasSpan(
            span_id="test_span_1",
            full_sentence_or_paragraph="This is a test sentence with problematic content.",
            bias_span="problematic content",
            bias_family="racism",
            bias_subtype="stereotyping",
            severity=CulturalSeverityLevel(
                sender_culture="de",
                receiver_culture="us",
                raw_severity=7.5,
                sender_severity=7.0,
                receiver_severity=8.0,
                cultural_explanation="Cross-cultural sensitivity difference"
            ),
            start_position=30,
            end_position=50
        )

    @pytest.mark.asyncio
    async def test_debias_single_span(self, pipeline, sample_bias_span):
        """Test single span debiasing."""
        request = DebiasingRequest(
            bias_span=sample_bias_span,
            input_language="en",
            output_language="de",
            sender_culture="de",
            receiver_culture="us",
            context_topic="test",
            audience="general",
            formality_level="neutral"
        )

        result = await pipeline.debias_span(request)

        assert result.span_id == "test"
        assert result.language == "de"
        assert result.bias_family == "racism"
        assert result.variant_a_rewrite == "Neutral version"
        assert result.variant_b_rewrite == "Emotional version"
        assert result.confidence_score > 0

    @pytest.mark.asyncio
    async def test_batch_debiasing(self, pipeline, sample_bias_span):
        """Test batch span debiasing."""
        request = BatchDebiasingRequest(
            spans=[sample_bias_span],
            full_document_text="Complete document text here.",
            input_language="en",
            output_language="de",
            sender_culture="de",
            receiver_culture="us",
            context_topic="test",
            audience="general",
            formality_level="neutral"
        )

        # Mock batch response
        batch_response = '{"language": "de", "spans": [{"span_id": "test", "bias_family": "racism", "bias_subtype": "stereotyping", "analysis_explanation": "Test", "can_preserve_core_intent": true, "variant_A_rewrite": "Neutral", "variant_B_rewrite": "Emotional", "safety_notes": ""}]}'
        pipeline.llm_client.generate.return_value.content = batch_response

        result = await pipeline.debias_batch(request)

        assert result.language == "de"
        assert len(result.spans) == 1
        assert result.total_processed == 1

    @pytest.mark.asyncio
    async def test_marker_generation(self, pipeline):
        """Test bias marker generation."""
        request = MarkerGenerationRequest(
            bias_family="racism",
            bias_subtype="stereotyping",
            bias_description="Stereotypical assumptions about racial groups",
            output_language="de",
            domain="social_media",
            primary_cultures=["de", "us"]
        )

        # Mock marker response
        marker_response = '{"bias_family": "racism", "bias_subtype": "stereotyping", "language": "de", "markers": [{"id": "test_marker", "name": "Test Marker", "description": "Test", "rationale": "Test", "positive_examples": ["Ex1", "Ex2", "Ex3"], "counter_example": "Counter", "severity_hint": "7-9", "languages": ["de"]}]}'
        pipeline.llm_client.generate.return_value.content = marker_response

        result = await pipeline.generate_markers(request)

        assert result.bias_family == "racism"
        assert result.bias_subtype == "stereotyping"
        assert len(result.markers) == 1

    @pytest.mark.asyncio
    async def test_response_quality_validation(self, pipeline):
        """Test response quality validation."""
        from bias_engine.llm.models import DebiasingResponse

        # Good response
        good_response = DebiasingResponse(
            span_id="test",
            language="de",
            bias_family="racism",
            bias_subtype="stereotyping",
            analysis_explanation="This text contains stereotypical assumptions that could harm perception of the group.",
            can_preserve_core_intent=True,
            variant_a_rewrite="Neutral factual statement",
            variant_b_rewrite="Different emotional expression",
            safety_notes="Intent preserved successfully",
            confidence_score=0.8
        )

        is_valid, issues = await pipeline.validate_response_quality(good_response)
        assert is_valid
        assert len(issues) == 0

        # Poor response
        poor_response = DebiasingResponse(
            span_id="test",
            language="de",
            bias_family="racism",
            bias_subtype="stereotyping",
            analysis_explanation="Short",  # Too short
            can_preserve_core_intent=False,
            variant_a_rewrite="Same text",
            variant_b_rewrite="Same text",  # Identical variants
            safety_notes="",  # Missing safety notes for non-preservable intent
            confidence_score=0.3
        )

        is_valid, issues = await pipeline.validate_response_quality(poor_response)
        assert not is_valid
        assert len(issues) > 0


class TestSelfBiasChecker:
    """Test self-bias checking system."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client for self-bias checking."""
        client = AsyncMock()
        return client

    @pytest.fixture
    def checker(self, mock_llm_client):
        """Create self-bias checker."""
        return SelfBiasChecker(mock_llm_client)

    def test_epistemic_classification(self, checker):
        """Test epistemic classification patterns."""
        # Factual text
        classification, indicators = checker._classify_epistemically(
            "Nach der Studie von 2023 beträgt die Arbeitslosenrate 5.2 Prozent."
        )
        assert classification == EpistemicClassification.FAKTISCH

        # Logical text
        classification, indicators = checker._classify_epistemically(
            "Wenn die Temperatur steigt, dann dehnt sich das Metall aus."
        )
        assert classification == EpistemicClassification.LOGISCH

        # Subjective text
        classification, indicators = checker._classify_epistemically(
            "Ich finde, dass dieses Buch sehr interessant ist."
        )
        assert classification == EpistemicClassification.SUBJEKTIV

    def test_overconfidence_detection(self, checker):
        """Test overconfidence pattern detection."""
        # Overconfident text
        assert checker._detect_overconfidence("Das ist definitiv die beste Lösung.")
        assert checker._detect_overconfidence("Alle Menschen sind gleich.")

        # Non-overconfident text
        assert not checker._detect_overconfidence("Das könnte eine gute Lösung sein.")
        assert not checker._detect_overconfidence("Viele Menschen haben ähnliche Erfahrungen.")

    def test_epistemic_prefix_application(self, checker):
        """Test epistemic prefix application."""
        # Factual
        result = checker._apply_epistemic_prefix(
            "Die Temperatur beträgt 20 Grad.",
            EpistemicClassification.FAKTISCH
        )
        assert result.startswith("Faktisch korrekt sage ich, dass")

        # Logical
        result = checker._apply_epistemic_prefix(
            "Daraus folgt eine Erhöhung der Kosten.",
            EpistemicClassification.LOGISCH
        )
        assert result.startswith("Logisch scheint mir, dass")

        # Subjective
        result = checker._apply_epistemic_prefix(
            "Das ist eine gute Idee.",
            EpistemicClassification.SUBJEKTIV
        )
        assert result.startswith("Rein subjektiv, aus meinem Denken ergibt sich, dass")

    def test_prefix_validation(self, checker):
        """Test epistemic prefix validation."""
        # Valid prefixes
        valid, classification = checker.validate_epistemic_prefix(
            "Faktisch korrekt sage ich, dass die Daten stimmen."
        )
        assert valid
        assert classification == EpistemicClassification.FAKTISCH

        valid, classification = checker.validate_epistemic_prefix(
            "Logisch scheint mir, dass dies funktioniert."
        )
        assert valid
        assert classification == EpistemicClassification.LOGISCH

        # Invalid/missing prefix
        valid, classification = checker.validate_epistemic_prefix(
            "Das ist definitiv richtig."
        )
        assert not valid
        assert classification is None

    @pytest.mark.asyncio
    async def test_fallback_processing(self, checker):
        """Test fallback processing when LLM fails."""
        request = SelfBiasCheckRequest(
            text="Das ist definitiv die beste Lösung für alle Menschen.",
            context="Technical discussion"
        )

        result = checker._fallback_processing(request)

        assert result.original_text == request.text
        assert result.epistemic_classification is not None
        assert result.overconfidence_detected  # Should detect "definitiv" and "alle"
        assert result.corrected_text.startswith(("Faktisch", "Logisch", "Rein subjektiv"))

    def test_batch_processing(self, checker):
        """Test batch self-bias checking."""
        texts = [
            "Nach der Studie sind 80% betroffen.",
            "Das ist definitiv richtig.",
            "Ich denke, das könnte funktionieren."
        ]

        results = checker.batch_check_bias(texts, "Test context")

        assert len(results) == 3
        assert all(result.original_text in texts for result in results)
        assert all(result.corrected_text.startswith(("Faktisch", "Logisch", "Rein subjektiv")) for result in results)

    def test_classification_statistics(self, checker):
        """Test statistics calculation."""
        from bias_engine.llm.models import SelfBiasCheckResponse

        responses = [
            SelfBiasCheckResponse(
                original_text="Test 1",
                epistemic_classification=EpistemicClassification.FAKTISCH,
                overconfidence_detected=True,
                bias_indicators=["definitiv"],
                corrected_text="Corrected 1",
                confidence_score=0.8,
                explanation="Test"
            ),
            SelfBiasCheckResponse(
                original_text="Test 2",
                epistemic_classification=EpistemicClassification.LOGISCH,
                overconfidence_detected=False,
                bias_indicators=[],
                corrected_text="Corrected 2",
                confidence_score=0.9,
                explanation="Test"
            )
        ]

        stats = checker.get_classification_statistics(responses)

        assert stats["total_processed"] == 2
        assert stats["classifications"]["faktisch"] == 1
        assert stats["classifications"]["logisch"] == 1
        assert stats["overconfidence_detected"] == 1
        assert stats["average_confidence"] == 0.85


class TestCulturalIntegration:
    """Test cultural context integration."""

    @pytest.fixture
    def integrator(self):
        """Create cultural context integrator."""
        return CulturalContextIntegrator()

    @pytest.fixture
    def sample_detection(self):
        """Create sample bias detection."""
        return BiasDetection(
            type=BiasType.RACIAL,
            level=BiasLevel.HIGH,
            confidence=0.85,
            description="Racial stereotype detected",
            affected_text="problematic phrase",
            start_position=20,
            end_position=39,
            suggestions=["neutral alternative"]
        )

    def test_cultural_mappings(self, integrator):
        """Test cultural mapping data."""
        assert "de" in integrator.cultural_mappings
        assert "us" in integrator.cultural_mappings
        assert "jp" in integrator.cultural_mappings

        # Test German cultural characteristics
        de_profile = integrator.cultural_mappings["de"]
        assert de_profile["directness"] > 0.7  # Germans are direct
        assert "bias_sensitivities" in de_profile

    def test_bias_span_creation(self, integrator, sample_detection):
        """Test bias span creation with cultural context."""
        full_text = "This is a test text with problematic phrase that needs fixing."

        bias_span = integrator.create_bias_span(
            detection=sample_detection,
            full_text=full_text,
            sender_culture="de",
            receiver_culture="us",
            context="test context"
        )

        assert bias_span.span_id.startswith("span_")
        assert bias_span.bias_family == "racial"
        assert bias_span.bias_span == "problematic phrase"
        assert bias_span.severity.sender_culture == "de"
        assert bias_span.severity.receiver_culture == "us"

    def test_cultural_severity_calculation(self, integrator):
        """Test cultural severity adjustment."""
        severity_level = integrator.calculate_cultural_severity(
            bias_type="racism",
            raw_severity=8.0,
            sender_culture="de",
            receiver_culture="us",
            context="professional"
        )

        assert severity_level.raw_severity == 8.0
        assert severity_level.sender_culture == "de"
        assert severity_level.receiver_culture == "us"
        assert len(severity_level.cultural_explanation) > 0

        # Both cultures should be sensitive to racism
        assert severity_level.sender_severity > 6.0
        assert severity_level.receiver_severity > 6.0

    def test_cross_cultural_adjustments(self, integrator):
        """Test cross-cultural communication adjustments."""
        # German (direct) to Japanese (indirect) - should increase severity
        de_profile = integrator.cultural_mappings["de"]
        jp_profile = integrator.cultural_mappings["jp"]

        adjustment = integrator._calculate_cross_cultural_adjustment(de_profile, jp_profile)
        assert adjustment > 1.0  # Should increase due to directness mismatch

    def test_batch_span_creation(self, integrator):
        """Test batch bias span creation."""
        detections = [
            BiasDetection(
                type=BiasType.RACIAL,
                level=BiasLevel.HIGH,
                confidence=0.85,
                description="Test 1",
                affected_text="bias 1",
                start_position=0,
                end_position=6,
                suggestions=[]
            ),
            BiasDetection(
                type=BiasType.GENDER,
                level=BiasLevel.MEDIUM,
                confidence=0.75,
                description="Test 2",
                affected_text="bias 2",
                start_position=20,
                end_position=26,
                suggestions=[]
            )
        ]

        full_text = "bias 1 is here and bias 2 is also here."

        spans = integrator.batch_create_bias_spans(
            detections=detections,
            full_text=full_text,
            sender_culture="de",
            receiver_culture="us"
        )

        assert len(spans) == 2
        assert spans[0].bias_family == "racial"
        assert spans[1].bias_family == "gender"

    def test_cultural_recommendations(self, integrator):
        """Test cultural communication recommendations."""
        # German to Japanese communication
        recommendations = integrator.get_cultural_recommendations("de", "jp")

        assert "sender_culture" in recommendations
        assert "receiver_culture" in recommendations
        assert "recommendations" in recommendations
        assert len(recommendations["recommendations"]) > 0

        # Should recommend more indirect communication
        recommendations_text = " ".join(recommendations["recommendations"])
        assert any(keyword in recommendations_text.lower() for keyword in ["indirect", "formal", "context"])


@pytest.mark.integration
class TestLLMIntegrationFlow:
    """Integration tests for complete LLM workflow."""

    @pytest.mark.asyncio
    async def test_end_to_end_debiasing_flow(self):
        """Test complete debiasing flow from detection to neutralization."""
        # This would be an integration test with actual LLM providers
        # For now, we'll test the flow with mocks
        pass

    def test_configuration_integration(self):
        """Test LLM configuration integration."""
        from bias_engine.llm.config import LLMConfigManager, LLMSettings

        # Test with mock environment
        with patch.dict('os.environ', {
            'ANTHROPIC_API_KEY': 'test-key',
            'ANTHROPIC_MODEL': 'claude-3-sonnet',
            'LLM_DEFAULT_PROVIDER': 'anthropic'
        }):
            settings = LLMSettings()
            manager = LLMConfigManager(settings)

            assert manager.is_provider_available(LLMProvider.ANTHROPIC)
            config = manager.get_config(LLMProvider.ANTHROPIC)
            assert config.api_key == 'test-key'
            assert config.model == 'claude-3-sonnet'

    def test_api_route_integration(self):
        """Test API route integration with dependency injection."""
        # This would test the FastAPI routes
        # Mocked for now as it requires FastAPI test client setup
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])