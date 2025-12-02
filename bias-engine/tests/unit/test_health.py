"""
Unit tests for health endpoints.
"""

import pytest
from fastapi import status


class TestHealthEndpoints:
    """Test cases for health check endpoints."""

    def test_health_check_success(self, client):
        """Test successful health check."""
        response = client.get("/api/v1/health")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "uptime" in data
        assert "dependencies" in data

        # Check status is either healthy or degraded
        assert data["status"] in ["healthy", "degraded"]

        # Check uptime is a positive number
        assert isinstance(data["uptime"], (int, float))
        assert data["uptime"] >= 0

        # Check dependencies structure
        assert isinstance(data["dependencies"], dict)

    def test_readiness_check_success(self, client):
        """Test readiness check endpoint."""
        response = client.get("/api/v1/health/ready")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] in ["ready", "not_ready"]

    def test_liveness_check_success(self, client):
        """Test liveness check endpoint."""
        response = client.get("/api/v1/health/live")

        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] == "alive"