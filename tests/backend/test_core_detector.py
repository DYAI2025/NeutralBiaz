"""
Unit tests for the core bias detection engine.
Tests all major bias detection algorithms and scoring mechanisms.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from typing import List, Dict, Any

# Test imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'bias-engine', 'src'))

from bias_engine.core.detector import BiasDetector
from bias_engine.core.models import BiasResult, BiasCategory, AnalysisRequest
from bias_engine.core.exceptions import BiasDetectionError, ValidationError


class TestCoreDetector:
    """Test suite for core bias detection functionality."""

    @pytest.fixture
    async def detector(self):
        """Create a bias detector instance for testing."""
        detector = BiasDetector()
        await detector.initialize()
        return detector

    @pytest.fixture
    def sample_texts(self) -> Dict[str, List[str]]:
        """Sample texts for different bias types."""
        return {
            "confirmation_bias": [
                "This research only confirms what we already knew to be true.",
                "Studies that contradict our findings are clearly flawed.",
                "The evidence supports our predetermined conclusion perfectly."
            ],
            "anchoring_bias": [
                "The first price mentioned seems reasonable, so all others should be compared to it.",
                "Based on the initial estimate, everything else looks expensive.",
                "Our original assessment was correct, so we should stick with it."
            ],
            "availability_bias": [
                "Since we recently heard about this issue, it must be very common.",
                "The vivid examples we remember make this seem like a major problem.",
                "This happened to someone I know, so it's definitely widespread."
            ],
            "neutral": [
                "The research methodology follows standard scientific protocols.",
                "Multiple data sources were consulted for comprehensive analysis.",
                "Results are presented objectively with acknowledged limitations."
            ]
        }

    @pytest.mark.unit
    async def test_initialize_detector(self):
        """Test that detector initializes properly with all components."""
        detector = BiasDetector()
        assert not detector.is_initialized

        await detector.initialize()
        assert detector.is_initialized
        assert detector.nlp_pipeline is not None
        assert detector.ml_classifier is not None
        assert detector.rule_engine is not None

    @pytest.mark.unit
    async def test_detect_confirmation_bias(self, detector, sample_texts):
        """Test detection of confirmation bias patterns."""
        for text in sample_texts["confirmation_bias"]:
            request = AnalysisRequest(text=text, cultural_context="en-US")
            result = await detector.analyze(request)

            assert isinstance(result, BiasResult)
            assert result.overall_score > 0.5
            assert BiasCategory.CONFIRMATION_BIAS in [b.category for b in result.detected_biases]
            assert result.confidence >= 0.7

    @pytest.mark.unit
    async def test_detect_anchoring_bias(self, detector, sample_texts):
        """Test detection of anchoring bias patterns."""
        for text in sample_texts["anchoring_bias"]:
            request = AnalysisRequest(text=text, cultural_context="en-US")
            result = await detector.analyze(request)

            assert result.overall_score > 0.4
            assert any(b.category == BiasCategory.ANCHORING_BIAS for b in result.detected_biases)

    @pytest.mark.unit
    async def test_detect_availability_bias(self, detector, sample_texts):
        """Test detection of availability bias patterns."""
        for text in sample_texts["availability_bias"]:
            request = AnalysisRequest(text=text, cultural_context="en-US")
            result = await detector.analyze(request)

            assert result.overall_score > 0.4
            assert any(b.category == BiasCategory.AVAILABILITY_BIAS for b in result.detected_biases)

    @pytest.mark.unit
    async def test_neutral_text_detection(self, detector, sample_texts):
        """Test that neutral text receives low bias scores."""
        for text in sample_texts["neutral"]:
            request = AnalysisRequest(text=text, cultural_context="en-US")
            result = await detector.analyze(request)

            assert result.overall_score < 0.3
            assert len(result.detected_biases) <= 1
            if result.detected_biases:
                assert result.detected_biases[0].severity == "low"

    @pytest.mark.unit
    async def test_score_aggregation(self, detector):
        """Test that bias scores are aggregated correctly."""
        text = "This clearly biased statement shows obvious confirmation and anchoring bias patterns."
        request = AnalysisRequest(text=text, cultural_context="en-US")
        result = await detector.analyze(request)

        individual_scores = [b.score for b in result.detected_biases]
        assert len(individual_scores) > 0

        # Overall score should be within reasonable range of individual scores
        max_individual = max(individual_scores) if individual_scores else 0
        assert result.overall_score <= max_individual + 0.1
        assert result.overall_score >= max(individual_scores) * 0.8 if individual_scores else True

    @pytest.mark.unit
    async def test_confidence_calculation(self, detector, sample_texts):
        """Test confidence score calculation for different bias types."""
        high_confidence_text = sample_texts["confirmation_bias"][0]
        request = AnalysisRequest(text=high_confidence_text, cultural_context="en-US")
        result = await detector.analyze(request)

        assert result.confidence >= 0.6
        assert all(b.confidence >= 0.5 for b in result.detected_biases)

    @pytest.mark.unit
    async def test_cultural_context_impact(self, detector):
        """Test that cultural context affects bias detection."""
        text = "Family honor is more important than individual preferences."

        # Test with different cultural contexts
        western_request = AnalysisRequest(text=text, cultural_context="en-US")
        eastern_request = AnalysisRequest(text=text, cultural_context="ja-JP")

        western_result = await detector.analyze(western_request)
        eastern_result = await detector.analyze(eastern_request)

        # Results should differ based on cultural context
        assert western_result.overall_score != eastern_result.overall_score

    @pytest.mark.unit
    async def test_empty_text_handling(self, detector):
        """Test handling of empty or whitespace-only text."""
        empty_texts = ["", "   ", "\n\t", None]

        for text in empty_texts:
            with pytest.raises(ValidationError):
                request = AnalysisRequest(text=text, cultural_context="en-US")
                await detector.analyze(request)

    @pytest.mark.unit
    async def test_very_long_text_handling(self, detector):
        """Test handling of very long text inputs."""
        long_text = "This is a test sentence. " * 1000  # ~25,000 characters
        request = AnalysisRequest(text=long_text, cultural_context="en-US")

        result = await detector.analyze(request)
        assert isinstance(result, BiasResult)
        assert result.processing_time < 30.0  # Should complete within 30 seconds

    @pytest.mark.unit
    async def test_bias_category_coverage(self, detector):
        """Test that all major bias categories can be detected."""
        bias_examples = {
            BiasCategory.CONFIRMATION_BIAS: "This only proves what I already believed.",
            BiasCategory.ANCHORING_BIAS: "Based on the first number I heard, this seems right.",
            BiasCategory.AVAILABILITY_BIAS: "I saw this on the news recently, so it must be common.",
            BiasCategory.SURVIVORSHIP_BIAS: "All successful people worked hard, so hard work guarantees success.",
            BiasCategory.SELECTION_BIAS: "We only surveyed our satisfied customers for feedback.",
        }

        detected_categories = set()
        for category, text in bias_examples.items():
            request = AnalysisRequest(text=text, cultural_context="en-US")
            result = await detector.analyze(request)

            for bias in result.detected_biases:
                detected_categories.add(bias.category)

        # Should detect at least 3 different bias categories
        assert len(detected_categories) >= 3

    @pytest.mark.unit
    async def test_concurrent_analysis(self, detector, sample_texts):
        """Test concurrent analysis of multiple texts."""
        texts = [text for texts_list in sample_texts.values() for text in texts_list]
        requests = [AnalysisRequest(text=text, cultural_context="en-US") for text in texts[:5]]

        # Run concurrent analyses
        tasks = [detector.analyze(request) for request in requests]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(isinstance(result, BiasResult) for result in results)
        assert all(result.processing_time < 10.0 for result in results)

    @pytest.mark.unit
    async def test_error_handling(self, detector):
        """Test error handling for various edge cases."""
        # Test with malformed request
        with pytest.raises(ValidationError):
            await detector.analyze(None)

        # Test with invalid cultural context
        request = AnalysisRequest(text="Test text", cultural_context="invalid-code")
        result = await detector.analyze(request)  # Should use default context
        assert isinstance(result, BiasResult)

    @pytest.mark.unit
    async def test_performance_benchmarks(self, detector):
        """Test performance benchmarks for bias detection."""
        test_text = "This is a moderate length text that contains some potential bias indicators and should be processed efficiently by the detection system."
        request = AnalysisRequest(text=test_text, cultural_context="en-US")

        # Measure processing time
        import time
        start_time = time.time()
        result = await detector.analyze(request)
        end_time = time.time()

        processing_time = end_time - start_time

        assert processing_time < 5.0  # Should complete within 5 seconds
        assert result.processing_time <= processing_time + 0.1
        assert result.processing_time > 0

    @pytest.mark.unit
    async def test_memory_usage(self, detector, sample_texts):
        """Test memory usage during analysis."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process multiple texts
        for text_list in sample_texts.values():
            for text in text_list:
                request = AnalysisRequest(text=text, cultural_context="en-US")
                await detector.analyze(request)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 100MB for this test)
        assert memory_increase < 100

    @pytest.mark.unit
    def test_bias_result_serialization(self):
        """Test that BiasResult objects can be properly serialized."""
        from bias_engine.core.models import DetectedBias

        detected_bias = DetectedBias(
            category=BiasCategory.CONFIRMATION_BIAS,
            score=0.75,
            confidence=0.85,
            evidence=["keyword match", "pattern detection"],
            severity="high",
            description="Strong confirmation bias detected"
        )

        result = BiasResult(
            text="Test text",
            overall_score=0.75,
            confidence=0.85,
            detected_biases=[detected_bias],
            cultural_context="en-US",
            processing_time=1.23,
            metadata={"model_version": "1.0"}
        )

        # Test JSON serialization
        result_dict = result.model_dump()
        assert "overall_score" in result_dict
        assert result_dict["overall_score"] == 0.75
        assert len(result_dict["detected_biases"]) == 1

    @pytest.mark.integration
    async def test_end_to_end_workflow(self, detector):
        """Test complete end-to-end bias detection workflow."""
        # Simulate a complete analysis workflow
        analysis_text = """
        Based on recent news coverage, it's clear that this problem is widespread
        and affects everyone. Our initial assessment was correct, and any studies
        that contradict our findings are obviously flawed. The solution we
        proposed first is clearly the best option.
        """

        request = AnalysisRequest(
            text=analysis_text,
            cultural_context="en-US",
            include_suggestions=True,
            detail_level="high"
        )

        result = await detector.analyze(request)

        # Verify comprehensive results
        assert result.overall_score > 0.6  # Should detect multiple biases
        assert len(result.detected_biases) >= 2  # Should find multiple bias types
        assert result.confidence > 0.7  # Should be confident in detection
        assert result.suggestions is not None  # Should provide suggestions
        assert len(result.suggestions) > 0

        # Verify bias categories detected
        detected_categories = [b.category for b in result.detected_biases]
        expected_biases = [
            BiasCategory.CONFIRMATION_BIAS,
            BiasCategory.AVAILABILITY_BIAS,
            BiasCategory.ANCHORING_BIAS
        ]

        # Should detect at least 2 of the expected biases
        matches = sum(1 for bias in expected_biases if bias in detected_categories)
        assert matches >= 2


# Performance benchmark tests
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmark tests for bias detection system."""

    @pytest.mark.asyncio
    async def test_throughput_benchmark(self):
        """Test system throughput with multiple concurrent requests."""
        detector = BiasDetector()
        await detector.initialize()

        texts = [
            "This confirms our existing beliefs perfectly.",
            "Based on the first estimate, everything else seems expensive.",
            "I saw this on the news, so it must be very common.",
            "All successful people work hard, so hard work guarantees success.",
        ] * 25  # 100 texts total

        import time
        start_time = time.time()

        # Process in batches of 10
        batch_size = 10
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            requests = [AnalysisRequest(text=text, cultural_context="en-US") for text in batch]
            batch_tasks = [detector.analyze(request) for request in requests]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)

        end_time = time.time()
        total_time = end_time - start_time
        throughput = len(texts) / total_time

        print(f"Processed {len(texts)} texts in {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} texts/second")

        # Performance targets
        assert throughput >= 5.0  # Should process at least 5 texts per second
        assert all(isinstance(result, BiasResult) for result in results)

    @pytest.mark.asyncio
    async def test_latency_benchmark(self):
        """Test response latency for individual requests."""
        detector = BiasDetector()
        await detector.initialize()

        test_texts = [
            "Short text.",
            "This is a medium length text with some bias indicators that should be detected by the system.",
            "This is a much longer text that contains multiple sentences and various types of potential cognitive biases including confirmation bias where we only look for evidence that supports our existing beliefs, anchoring bias where we rely too heavily on the first piece of information encountered, and availability bias where we overestimate the importance of information that comes easily to mind, and we need to test how well the system performs with longer input texts."
        ]

        latencies = []

        for text in test_texts:
            request = AnalysisRequest(text=text, cultural_context="en-US")

            import time
            start_time = time.time()
            result = await detector.analyze(request)
            end_time = time.time()

            latency = end_time - start_time
            latencies.append(latency)

            print(f"Text length: {len(text)}, Latency: {latency:.3f}s")

        # Performance targets
        assert max(latencies) < 10.0  # No request should take more than 10 seconds
        assert np.mean(latencies) < 3.0  # Average latency should be under 3 seconds

        # Verify results quality wasn't sacrificed for speed
        for text in test_texts:
            request = AnalysisRequest(text=text, cultural_context="en-US")
            result = await detector.analyze(request)
            assert isinstance(result, BiasResult)
            assert 0.0 <= result.overall_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])