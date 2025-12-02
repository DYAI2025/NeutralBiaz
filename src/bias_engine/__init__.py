#!/usr/bin/env python3
"""
Bias Engine Package

Core bias detection functionality.
"""

from .core_detector import (
    detect_bias_spans,
    classify_bias_type,
    calculate_severity,
    calculate_confidence,
    get_detection_engine,
    DetectionConfig,
    BiasDetectionEngine
)

from .taxonomy_loader import (
    get_taxonomy_loader,
    reload_taxonomy,
    BiaxTaxonomyLoader
)

from .nlp_pipeline import (
    get_nlp_pipeline,
    NLPPipeline,
    LanguageDetector,
    TextPreprocessor
)

from .rule_based_detector import (
    get_rule_based_detector,
    RuleBasedBiasDetector
)

from .ml_classifier import (
    get_ml_classifier,
    EnsembleBiasClassifier
)

from .scoring_algorithms import (
    get_confidence_calculator,
    get_severity_calculator,
    get_aggregated_scoring,
    ConfidenceCalculator,
    SeverityCalculator
)

from .config_manager import (
    get_config_manager,
    create_default_config,
    ConfigurationManager,
    BiasEngineConfig
)

__all__ = [
    # Main API functions
    'detect_bias_spans',
    'classify_bias_type',
    'calculate_severity',
    'calculate_confidence',
    'get_detection_engine',
    
    # Configuration
    'DetectionConfig',
    'BiasEngineConfig',
    'get_config_manager',
    'create_default_config',
    
    # Core classes
    'BiasDetectionEngine',
    'ConfigurationManager',
    
    # Component getters
    'get_taxonomy_loader',
    'get_nlp_pipeline',
    'get_rule_based_detector',
    'get_ml_classifier',
    'get_confidence_calculator',
    'get_severity_calculator',
    'get_aggregated_scoring',
    
    # Utility functions
    'reload_taxonomy',
    'create_default_config'
]
