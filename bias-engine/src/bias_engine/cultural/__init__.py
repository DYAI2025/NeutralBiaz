"""
Cultural Adaptation Engine for Bias Analysis

This module provides cultural dimension analysis based on Hofstede's 6D model
for enhancing bias detection with cross-cultural intelligence.
"""

from .models.hofstede_model import HofstedeModel, CulturalDimensions
from .adapters.cultural_adapter import CulturalAdapter
from .analyzers.cultural_analyzer import CulturalAnalyzer
from .intelligence.cultural_intelligence import CulturalIntelligence
from .data.profile_manager import CulturalProfileManager

__version__ = "1.0.0"

__all__ = [
    "HofstedeModel",
    "CulturalDimensions",
    "CulturalAdapter",
    "CulturalAnalyzer",
    "CulturalIntelligence",
    "CulturalProfileManager"
]