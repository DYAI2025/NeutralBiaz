"""
Unit tests for analysis endpoints.
"""

import pytest
from fastapi import status


class TestAnalyzeEndpoints:
    """Test cases for text analysis endpoints."""

    def test_analyze_text_success(self, client, sample_analysis_request):
        """Test successful text analysis."""
        response = client.post("/api/v1/analyze", json=sample_analysis_request)

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "text" in data
        assert "language" in data
        assert "overall_bias_score" in data
        assert "bias_level" in data
        assert "detections" in data
        assert "processing_time" in data
        assert "model_version" in data

        # Check data types
        assert isinstance(data["overall_bias_score"], (int, float))
        assert 0 <= data["overall_bias_score"] <= 1
        assert data["bias_level"] in ["low", "medium", "high", "critical"]
        assert isinstance(data["detections"], list)
        assert isinstance(data["processing_time"], (int, float))
        assert data["processing_time"] >= 0

    def test_analyze_text_empty_text(self, client):
        """Test analysis with empty text."""
        request_data = {
            "text": "",
            "mode": "fast",
        }

        response = client.post("/api/v1/analyze", json=request_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_analyze_text_whitespace_only(self, client):
        """Test analysis with whitespace-only text."""
        request_data = {
            "text": "   ",
            "mode": "fast",
        }

        response = client.post("/api/v1/analyze", json=request_data)

        # This should fail validation since we strip whitespace
        assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_422_UNPROCESSABLE_ENTITY]

    def test_analyze_text_long_text(self, client):
        """Test analysis with very long text."""
        request_data = {
            "text": "A" * 15000,  # Exceeds max length
            "mode": "fast",
        }

        response = client.post("/api/v1/analyze", json=request_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_analyze_batch_success(self, client, sample_batch_request):
        """Test successful batch analysis."""
        response = client.post("/api/v1/analyze/batch", json=sample_batch_request)

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "results" in data
        assert "total_processed" in data
        assert "total_processing_time" in data
        assert "summary" in data

        # Check results structure
        assert isinstance(data["results"], list)
        assert len(data["results"]) <= len(sample_batch_request["texts"])
        assert data["total_processed"] == len(data["results"])

        # Check each result has required fields
        for result in data["results"]:
            assert "text" in result
            assert "overall_bias_score" in result
            assert "bias_level" in result
            assert "detections" in result

        # Check summary structure
        summary = data["summary"]
        assert "total_detections" in summary
        assert "average_bias_score" in summary
        assert "texts_with_bias" in summary
        assert "success_rate" in summary

    def test_analyze_batch_empty_list(self, client):
        """Test batch analysis with empty texts list."""
        request_data = {
            "texts": [],
            "mode": "fast",
        }

        response = client.post("/api/v1/analyze/batch", json=request_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_analyze_batch_too_many_texts(self, client):
        """Test batch analysis with too many texts."""
        request_data = {
            "texts": ["Sample text"] * 150,  # Exceeds max items
            "mode": "fast",
        }

        response = client.post("/api/v1/analyze/batch", json=request_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_analyze_with_different_modes(self, client):
        """Test analysis with different modes."""
        modes = ["fast", "accurate", "comprehensive"]

        for mode in modes:
            request_data = {
                "text": "He should be the CEO.",
                "mode": mode,
            }

            response = client.post("/api/v1/analyze", json=request_data)
            assert response.status_code == status.HTTP_200_OK

    def test_analyze_with_cultural_profiles(self, client):
        """Test analysis with different cultural profiles."""
        profiles = ["neutral", "western", "eastern", "global"]

        for profile in profiles:
            request_data = {
                "text": "He should be the CEO.",
                "cultural_profile": profile,
            }

            response = client.post("/api/v1/analyze", json=request_data)
            assert response.status_code == status.HTTP_200_OK