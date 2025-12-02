#!/usr/bin/env python3
"""
Bias Detection Models Package

Data models and type definitions for bias detection.
"""

from .bias_models import (
    BiasSpan,
    BiasDetectionResult,
    BiasClassification,
    BiasFamily,
    BiasSubtype,
    IntersectionalAnalysis,
    DetectionMethod,
    BiasSeverityLevel,
    BiasConfidenceLevel,
    BiasPatternMatch,
    validate_bias_family_config,
    validate_bias_subtype_config
)

__all__ = [
    'BiasSpan',
    'BiasDetectionResult',
    'BiasClassification',
    'BiasFamily',
    'BiasSubtype',
    'IntersectionalAnalysis',
    'DetectionMethod',
    'BiasSeverityLevel',
    'BiasConfidenceLevel',
    'BiasPatternMatch',
    'validate_bias_family_config',
    'validate_bias_subtype_config'
]
