"""
Bias Engine - AI-powered bias detection and neutralization system.

This package provides tools for detecting and neutralizing bias in text content
using advanced NLP techniques and machine learning models.
"""

__version__ = "1.0.0"
__author__ = "BiazNeutralize AI Team"
__email__ = "team@biazneutralize.ai"

from .core.config import settings
from .core.exceptions import BiasEngineError

__all__ = ["settings", "BiasEngineError"]