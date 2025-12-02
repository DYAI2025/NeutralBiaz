"""
Unit tests for configuration endpoints.
"""

import pytest
from fastapi import status


class TestConfigEndpoints:
    """Test cases for configuration endpoints."""

    def test_get_config_success(self, client):
        """Test successful configuration retrieval."""
        response = client.get("/api/v1/config")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        required_fields = [
            "app_name",
            "app_version",
            "environment",
            "supported_languages",
            "supported_bias_types",
            "cultural_profiles",
            "analysis_modes",
            "default_settings"
        ]

        for field in required_fields:
            assert field in data

        # Check supported languages is a list
        assert isinstance(data["supported_languages"], list)
        assert len(data["supported_languages"]) > 0

        # Check supported bias types
        assert isinstance(data["supported_bias_types"], list)
        expected_bias_types = [
            "gender", "racial", "age", "religious", "political",
            "socioeconomic", "cultural", "disability", "lgbtq", "unknown"
        ]
        for bias_type in data["supported_bias_types"]:
            assert bias_type in expected_bias_types

        # Check cultural profiles
        assert isinstance(data["cultural_profiles"], list)
        expected_profiles = ["neutral", "western", "eastern", "global"]
        for profile in data["cultural_profiles"]:
            assert profile in expected_profiles

        # Check analysis modes
        assert isinstance(data["analysis_modes"], list)
        expected_modes = ["fast", "accurate", "comprehensive"]
        for mode in data["analysis_modes"]:
            assert mode in expected_modes

        # Check default settings structure
        assert isinstance(data["default_settings"], dict)
        default_settings = data["default_settings"]
        assert "bias_threshold" in default_settings
        assert "confidence_threshold" in default_settings
        assert "default_model" in default_settings

        # Validate threshold values
        assert 0 <= default_settings["bias_threshold"] <= 1
        assert 0 <= default_settings["confidence_threshold"] <= 1