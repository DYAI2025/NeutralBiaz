"""
Pytest configuration and fixtures.
"""

import pytest
from fastapi.testclient import TestClient

from bias_engine.main import app


@pytest.fixture
def client():
    """Create test client for FastAPI application."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return {
        "neutral": "The weather is nice today.",
        "gender_bias": "He should be the CEO because men are better leaders.",
        "age_bias": "Old people are not good with technology.",
        "racial_bias": "People from that country are always lazy.",
        "multiple_bias": "Young women should not work in technical fields.",
        "empty": "",
        "long": "A" * 10000,
    }


@pytest.fixture
def sample_analysis_request():
    """Sample analysis request payload."""
    return {
        "text": "He should be the CEO because men are better leaders.",
        "mode": "fast",
        "cultural_profile": "neutral",
        "language": "en",
        "confidence_threshold": 0.5,
        "include_suggestions": True,
    }


@pytest.fixture
def sample_batch_request():
    """Sample batch analysis request payload."""
    return {
        "texts": [
            "The weather is nice today.",
            "He should be the CEO because men are better leaders.",
            "Old people are not good with technology.",
        ],
        "mode": "fast",
        "cultural_profile": "neutral",
        "language": "en",
        "confidence_threshold": 0.5,
        "include_suggestions": True,
    }