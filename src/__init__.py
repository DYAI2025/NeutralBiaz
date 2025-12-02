#!/usr/bin/env python3
"""
Bias Detection Engine

A comprehensive bias detection system with intersectional taxonomy support.
"""

__version__ = "1.0.0"
__author__ = "BiazNeutralize AI Team"
__description__ = "Advanced bias detection engine with intersectional analysis"

from .bias_engine import (
    detect_bias_spans,
    classify_bias_type,
    calculate_severity,
    calculate_confidence,
    get_detection_engine,
    DetectionConfig
)

from .models.bias_models import (
    BiasSpan,
    BiasDetectionResult,
    BiasClassification,
    BiasFamily,
    BiasSubtype,
    IntersectionalAnalysis,
    DetectionMethod,
    BiasSeverityLevel,
    BiasConfidenceLevel
)

__all__ = [
    # Main functions
    'detect_bias_spans',
    'classify_bias_type', 
    'calculate_severity',
    'calculate_confidence',
    'get_detection_engine',
    
    # Configuration
    'DetectionConfig',
    
    # Models
    'BiasSpan',
    'BiasDetectionResult',
    'BiasClassification', 
    'BiasFamily',
    'BiasSubtype',
    'IntersectionalAnalysis',
    'DetectionMethod',
    'BiasSeverityLevel',
    'BiasConfidenceLevel',
    
    # Metadata
    '__version__',
    '__author__',
    '__description__'
]
