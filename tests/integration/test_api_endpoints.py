"""
Integration tests for API endpoints.
Tests complete API workflows, request/response handling, and end-to-end functionality.
"""
import pytest
import asyncio
import json
from typing import Dict, Any, List
import httpx
from unittest.mock import patch, Mock
import time

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'bias-engine', 'src'))

from fastapi.testclient import TestClient
from bias_engine.api.main import create_app
from bias_engine.core.models import AnalysisRequest, BiasResult
from bias_engine.api.models import (
    BiasAnalysisRequest,
    BiasAnalysisResponse,
    TextNeutralizationRequest,
    CulturalProfileRequest
)


@pytest.fixture(scope="session")
def test_app():
    """Create test application instance."""
    app = create_app()
    return app


@pytest.fixture(scope="session")
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


@pytest.fixture
def sample_requests() -> Dict[str, Dict[str, Any]]:
    """Sample API requests for testing."""
    return {
        "bias_analysis": {
            "text": "This evidence clearly proves that our hypothesis is correct and any contradictory studies are obviously flawed.",
            "cultural_context": "en-US",
            "include_suggestions": True,
            "detail_level": "high"
        },
        "text_neutralization": {
            "text": "This revolutionary new approach obviously outperforms all existing methods.",
            "cultural_context": "en-US",
            "preserve_style": True,
            "target_audience": "academic"
        },
        "cultural_profile": {
            "culture_codes": ["en-US", "ja-JP", "de-DE"],
            "include_comparisons": True
        },
        "batch_analysis": {
            "texts": [
                "The research clearly demonstrates our point.",
                "Based on the initial findings, all subsequent results support the theory.",
                "Recent media coverage suggests this is a widespread problem."
            ],
            "cultural_context": "en-US",
            "batch_size": 3
        }
    }


@pytest.fixture
def expected_responses() -> Dict[str, Dict[str, Any]]:
    """Expected API response structures."""
    return {
        "bias_analysis": {
            "overall_score": float,
            "confidence": float,
            "detected_biases": list,
            "cultural_context": str,
            "processing_time": float,
            "suggestions": list,
            "metadata": dict
        },
        "text_neutralization": {
            "original_text": str,
            "neutralized_text": str,
            "changes_made": list,
            "confidence": float,
            "cultural_adaptations": dict
        },
        "cultural_profile": {
            "profiles": list,
            "comparisons": dict
        }
    }


class TestBiasAnalysisEndpoint:
    """Test suite for bias analysis API endpoint."""

    @pytest.mark.integration
    def test_bias_analysis_basic_request(self, client, sample_requests, expected_responses):
        """Test basic bias analysis request."""
        request_data = sample_requests["bias_analysis"]

        response = client.post("/api/v1/analyze/bias", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        expected_fields = expected_responses["bias_analysis"]
        for field, field_type in expected_fields.items():
            assert field in data
            if field_type == float:
                assert isinstance(data[field], (int, float))
                assert 0.0 <= data[field] <= 1.0 if field in ["overall_score", "confidence"] else True
            elif field_type == list:
                assert isinstance(data[field], list)
            elif field_type == dict:
                assert isinstance(data[field], dict)

        # Verify bias detection results
        assert len(data["detected_biases"]) > 0
        for bias in data["detected_biases"]:
            assert "category" in bias
            assert "score" in bias
            assert "confidence" in bias
            assert "evidence" in bias

    @pytest.mark.integration
    def test_bias_analysis_with_cultural_context(self, client, sample_requests):
        """Test bias analysis with different cultural contexts."""
        base_request = sample_requests["bias_analysis"].copy()

        cultures = ["en-US", "ja-JP", "de-DE", "fr-FR"]
        results = {}

        for culture in cultures:
            base_request["cultural_context"] = culture
            response = client.post("/api/v1/analyze/bias", json=base_request)

            assert response.status_code == 200
            results[culture] = response.json()

        # Verify cultural differences
        scores = {culture: result["overall_score"] for culture, result in results.items()}

        # Should have some variation in scores based on cultural context
        score_range = max(scores.values()) - min(scores.values())
        assert score_range > 0.1  # Expect at least 0.1 difference

        # Verify cultural context is preserved
        for culture, result in results.items():
            assert result["cultural_context"] == culture

    @pytest.mark.integration
    def test_bias_analysis_detail_levels(self, client, sample_requests):
        """Test different detail levels in bias analysis."""
        base_request = sample_requests["bias_analysis"].copy()

        detail_levels = ["low", "medium", "high"]
        results = {}

        for level in detail_levels:
            base_request["detail_level"] = level
            response = client.post("/api/v1/analyze/bias", json=base_request)

            assert response.status_code == 200
            results[level] = response.json()

        # Higher detail levels should provide more information
        assert len(results["high"]["detected_biases"]) >= len(results["medium"]["detected_biases"])
        assert len(results["medium"]["detected_biases"]) >= len(results["low"]["detected_biases"])

        # High detail should include more evidence
        if results["high"]["detected_biases"]:
            high_evidence = results["high"]["detected_biases"][0].get("evidence", [])
            low_evidence = results["low"]["detected_biases"][0].get("evidence", []) if results["low"]["detected_biases"] else []
            assert len(high_evidence) >= len(low_evidence)

    @pytest.mark.integration
    def test_bias_analysis_with_suggestions(self, client, sample_requests):
        """Test bias analysis with improvement suggestions."""
        request_data = sample_requests["bias_analysis"].copy()
        request_data["include_suggestions"] = True

        response = client.post("/api/v1/analyze/bias", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Should include suggestions when requested
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)

        if data["overall_score"] > 0.3:  # If bias detected
            assert len(data["suggestions"]) > 0

            # Verify suggestion structure
            for suggestion in data["suggestions"]:
                assert "type" in suggestion
                assert "description" in suggestion
                assert "example" in suggestion

    @pytest.mark.integration
    def test_bias_analysis_error_handling(self, client):
        """Test error handling in bias analysis endpoint."""
        # Test empty text
        response = client.post("/api/v1/analyze/bias", json={
            "text": "",
            "cultural_context": "en-US"
        })
        assert response.status_code == 422

        # Test missing required fields
        response = client.post("/api/v1/analyze/bias", json={
            "cultural_context": "en-US"
        })
        assert response.status_code == 422

        # Test invalid cultural context
        response = client.post("/api/v1/analyze/bias", json={
            "text": "Test text",
            "cultural_context": "invalid-code"
        })
        assert response.status_code == 200  # Should use default context

        # Test extremely long text
        long_text = "This is a test. " * 10000  # Very long text
        response = client.post("/api/v1/analyze/bias", json={
            "text": long_text,
            "cultural_context": "en-US"
        })
        assert response.status_code in [200, 413]  # Either success or payload too large


class TestTextNeutralizationEndpoint:
    """Test suite for text neutralization API endpoint."""

    @pytest.mark.integration
    def test_text_neutralization_basic(self, client, sample_requests, expected_responses):
        """Test basic text neutralization request."""
        request_data = sample_requests["text_neutralization"]

        response = client.post("/api/v1/neutralize/text", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        expected_fields = expected_responses["text_neutralization"]
        for field, field_type in expected_fields.items():
            assert field in data
            if field_type == str:
                assert isinstance(data[field], str)
                assert len(data[field]) > 0
            elif field_type == list:
                assert isinstance(data[field], list)
            elif field_type == dict:
                assert isinstance(data[field], dict)

        # Verify neutralization occurred
        assert data["neutralized_text"] != data["original_text"]
        assert len(data["changes_made"]) > 0

    @pytest.mark.integration
    def test_text_neutralization_style_preservation(self, client, sample_requests):
        """Test text neutralization with style preservation."""
        base_request = sample_requests["text_neutralization"].copy()

        # Test with style preservation enabled
        base_request["preserve_style"] = True
        response_preserved = client.post("/api/v1/neutralize/text", json=base_request)

        # Test with style preservation disabled
        base_request["preserve_style"] = False
        response_not_preserved = client.post("/api/v1/neutralize/text", json=base_request)

        assert response_preserved.status_code == 200
        assert response_not_preserved.status_code == 200

        preserved_data = response_preserved.json()
        not_preserved_data = response_not_preserved.json()

        # Results should differ based on style preservation
        assert preserved_data["neutralized_text"] != not_preserved_data["neutralized_text"]

    @pytest.mark.integration
    def test_text_neutralization_target_audiences(self, client, sample_requests):
        """Test text neutralization for different target audiences."""
        base_request = sample_requests["text_neutralization"].copy()

        audiences = ["general", "academic", "business", "educational"]
        results = {}

        for audience in audiences:
            base_request["target_audience"] = audience
            response = client.post("/api/v1/neutralize/text", json=base_request)

            assert response.status_code == 200
            results[audience] = response.json()

        # Different audiences should produce different neutralizations
        academic_text = results["academic"]["neutralized_text"]
        general_text = results["general"]["neutralized_text"]

        # Academic version should be more formal
        assert "research" in academic_text.lower() or "study" in academic_text.lower() or len(academic_text) > len(general_text)


class TestCulturalProfileEndpoint:
    """Test suite for cultural profile API endpoint."""

    @pytest.mark.integration
    def test_cultural_profile_basic(self, client, sample_requests, expected_responses):
        """Test basic cultural profile request."""
        request_data = sample_requests["cultural_profile"]

        response = client.post("/api/v1/cultural/profiles", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        expected_fields = expected_responses["cultural_profile"]
        for field, field_type in expected_fields.items():
            assert field in data
            assert isinstance(data[field], field_type)

        # Verify profiles for requested cultures
        requested_cultures = request_data["culture_codes"]
        assert len(data["profiles"]) == len(requested_cultures)

        for profile in data["profiles"]:
            assert profile["culture_code"] in requested_cultures
            assert "individualism_score" in profile
            assert "power_distance" in profile
            assert "uncertainty_avoidance" in profile
            assert "bias_sensitivity" in profile

    @pytest.mark.integration
    def test_cultural_profile_comparisons(self, client, sample_requests):
        """Test cultural profile comparisons."""
        request_data = sample_requests["cultural_profile"].copy()
        request_data["include_comparisons"] = True

        response = client.post("/api/v1/cultural/profiles", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Should include comparisons when requested
        assert "comparisons" in data
        assert isinstance(data["comparisons"], dict)
        assert len(data["comparisons"]) > 0

        # Verify comparison structure
        for comparison_key, comparison_data in data["comparisons"].items():
            assert "cultures" in comparison_data
            assert "differences" in comparison_data
            assert "similarity_score" in comparison_data


class TestBatchProcessingEndpoint:
    """Test suite for batch processing API endpoint."""

    @pytest.mark.integration
    def test_batch_bias_analysis(self, client, sample_requests):
        """Test batch bias analysis."""
        request_data = sample_requests["batch_analysis"]

        response = client.post("/api/v1/analyze/batch", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Verify batch response structure
        assert "results" in data
        assert "batch_metadata" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) == len(request_data["texts"])

        # Verify individual results
        for i, result in enumerate(data["results"]):
            assert "text_index" in result
            assert result["text_index"] == i
            assert "overall_score" in result
            assert "detected_biases" in result
            assert "processing_time" in result

        # Verify batch metadata
        metadata = data["batch_metadata"]
        assert "total_texts" in metadata
        assert "total_processing_time" in metadata
        assert "average_processing_time" in metadata
        assert metadata["total_texts"] == len(request_data["texts"])

    @pytest.mark.integration
    def test_batch_processing_limits(self, client):
        """Test batch processing limits and error handling."""
        # Test with too many texts
        large_batch = {
            "texts": ["Test text"] * 1000,  # Assuming limit is lower
            "cultural_context": "en-US"
        }

        response = client.post("/api/v1/analyze/batch", json=large_batch)
        # Should either succeed or return appropriate error
        assert response.status_code in [200, 413, 422]

        if response.status_code != 200:
            error_data = response.json()
            assert "detail" in error_data

    @pytest.mark.integration
    def test_concurrent_batch_requests(self, client, sample_requests):
        """Test concurrent batch processing requests."""
        import asyncio
        import httpx

        async def make_batch_request():
            async with httpx.AsyncClient(app=client.app, base_url="http://test") as ac:
                response = await ac.post("/api/v1/analyze/batch", json=sample_requests["batch_analysis"])
                return response

        # Make multiple concurrent batch requests
        async def run_concurrent_tests():
            tasks = [make_batch_request() for _ in range(3)]
            responses = await asyncio.gather(*tasks)
            return responses

        responses = asyncio.run(run_concurrent_tests())

        # All should succeed
        for response in responses:
            assert response.status_code == 200


class TestAPIPerformance:
    """Performance tests for API endpoints."""

    @pytest.mark.slow
    def test_bias_analysis_performance(self, client, sample_requests):
        """Test bias analysis endpoint performance."""
        request_data = sample_requests["bias_analysis"]

        # Measure response time for multiple requests
        response_times = []
        num_requests = 10

        for _ in range(num_requests):
            start_time = time.time()
            response = client.post("/api/v1/analyze/bias", json=request_data)
            end_time = time.time()

            assert response.status_code == 200
            response_times.append(end_time - start_time)

        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)

        print(f"Average response time: {avg_response_time:.3f}s")
        print(f"Max response time: {max_response_time:.3f}s")

        # Performance targets
        assert avg_response_time < 5.0  # Average under 5 seconds
        assert max_response_time < 10.0  # No request over 10 seconds

    @pytest.mark.slow
    def test_api_throughput(self, client, sample_requests):
        """Test API throughput under load."""
        import concurrent.futures
        import threading
        from concurrent.futures import ThreadPoolExecutor

        def make_request():
            response = client.post("/api/v1/analyze/bias", json=sample_requests["bias_analysis"])
            return response.status_code == 200

        # Test concurrent requests
        num_concurrent = 20
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(num_concurrent)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        end_time = time.time()
        total_time = end_time - start_time
        throughput = num_concurrent / total_time

        print(f"Processed {num_concurrent} requests in {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} requests/second")

        # Performance targets
        assert throughput >= 2.0  # At least 2 requests per second
        assert sum(results) >= num_concurrent * 0.95  # 95% success rate

    @pytest.mark.slow
    def test_memory_usage_under_load(self, client, sample_requests):
        """Test memory usage during API load testing."""
        import psutil
        import gc

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Make many requests
        for _ in range(50):
            response = client.post("/api/v1/analyze/bias", json=sample_requests["bias_analysis"])
            assert response.status_code == 200

        # Force garbage collection
        gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"Memory increase: {memory_increase:.2f} MB")

        # Memory usage should be reasonable
        assert memory_increase < 200  # Less than 200MB increase


class TestAPIErrorHandling:
    """Test API error handling and edge cases."""

    @pytest.mark.integration
    def test_malformed_json_handling(self, client):
        """Test handling of malformed JSON requests."""
        # Test with malformed JSON
        response = client.post(
            "/api/v1/analyze/bias",
            data="{ invalid json }",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    @pytest.mark.integration
    def test_missing_content_type(self, client, sample_requests):
        """Test handling of requests with missing content type."""
        response = client.post(
            "/api/v1/analyze/bias",
            data=json.dumps(sample_requests["bias_analysis"])
            # No Content-Type header
        )
        # Should either work or return appropriate error
        assert response.status_code in [200, 415, 422]

    @pytest.mark.integration
    def test_unsupported_http_methods(self, client):
        """Test handling of unsupported HTTP methods."""
        # Test GET on POST endpoint
        response = client.get("/api/v1/analyze/bias")
        assert response.status_code == 405

        # Test PUT on POST endpoint
        response = client.put("/api/v1/analyze/bias", json={"text": "test"})
        assert response.status_code == 405

    @pytest.mark.integration
    def test_rate_limiting(self, client, sample_requests):
        """Test rate limiting behavior."""
        # Make many rapid requests
        responses = []
        for _ in range(100):
            response = client.post("/api/v1/analyze/bias", json=sample_requests["bias_analysis"])
            responses.append(response)

        status_codes = [r.status_code for r in responses]

        # Should mostly succeed, but might have some rate limiting
        success_rate = sum(1 for code in status_codes if code == 200) / len(status_codes)
        assert success_rate >= 0.8  # At least 80% should succeed

        # Check for rate limiting responses
        rate_limited = sum(1 for code in status_codes if code == 429)
        if rate_limited > 0:
            print(f"Rate limited {rate_limited} out of {len(responses)} requests")


class TestAPIVersioning:
    """Test API versioning and backward compatibility."""

    @pytest.mark.integration
    def test_api_version_headers(self, client, sample_requests):
        """Test API version handling."""
        # Test with version header
        response = client.post(
            "/api/v1/analyze/bias",
            json=sample_requests["bias_analysis"],
            headers={"API-Version": "1.0"}
        )
        assert response.status_code == 200

        # Check response includes version info
        assert "X-API-Version" in response.headers or "api_version" in response.json().get("metadata", {})

    @pytest.mark.integration
    def test_endpoint_documentation(self, client):
        """Test that API documentation endpoints work."""
        # Test OpenAPI/Swagger docs
        response = client.get("/docs")
        assert response.status_code == 200

        # Test API schema
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
        assert "/api/v1/analyze/bias" in schema["paths"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])