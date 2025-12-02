"""
Unit tests for models endpoints.
"""

import pytest
from fastapi import status


class TestModelsEndpoints:
    """Test cases for models information endpoints."""

    def test_get_models_success(self, client):
        """Test successful models retrieval."""
        response = client.get("/api/v1/models")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        required_fields = ["models", "default_model", "total_models"]

        for field in required_fields:
            assert field in data

        # Check models list structure
        assert isinstance(data["models"], list)
        assert data["total_models"] == len(data["models"])
        assert data["total_models"] > 0

        # Check each model has required fields
        for model in data["models"]:
            model_fields = [
                "name", "version", "description", "supported_languages",
                "supported_bias_types", "is_default", "loaded"
            ]
            for field in model_fields:
                assert field in model

            # Check data types
            assert isinstance(model["name"], str)
            assert isinstance(model["version"], str)
            assert isinstance(model["supported_languages"], list)
            assert isinstance(model["supported_bias_types"], list)
            assert isinstance(model["is_default"], bool)
            assert isinstance(model["loaded"], bool)

            # Check accuracy if present
            if "accuracy" in model and model["accuracy"] is not None:
                assert 0 <= model["accuracy"] <= 1

        # Check that exactly one model is marked as default
        default_models = [m for m in data["models"] if m["is_default"]]
        assert len(default_models) == 1

    def test_get_specific_model_success(self, client):
        """Test getting information about a specific model."""
        # First get all models to find a valid model name
        response = client.get("/api/v1/models")
        assert response.status_code == status.HTTP_200_OK

        models_data = response.json()
        assert len(models_data["models"]) > 0

        # Test getting the first model
        model_name = models_data["models"][0]["name"]
        response = client.get(f"/api/v1/models/{model_name}")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["name"] == model_name

        # Check required fields
        required_fields = [
            "name", "version", "description", "supported_languages",
            "supported_bias_types", "is_default", "loaded"
        ]
        for field in required_fields:
            assert field in data

    def test_get_nonexistent_model(self, client):
        """Test getting information about a non-existent model."""
        response = client.get("/api/v1/models/nonexistent-model")

        assert response.status_code == status.HTTP_404_NOT_FOUND

        data = response.json()
        assert "detail" in data
        assert "nonexistent-model" in data["detail"]