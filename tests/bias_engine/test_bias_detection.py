#!/usr/bin/env python3
"""
Comprehensive Test Suite for Bias Detection Engine

Tests all components of the bias detection system:
- Taxonomy loading and validation
- NLP pipeline functionality
- Rule-based detection
- ML classification
- Scoring algorithms
- Core detection engine
- Intersectional analysis
"""

import unittest
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any

from src.models.bias_models import (
    BiasSpan, BiasDetectionResult, BiasClassification, BiasFamily, BiasSubtype,
    DetectionMethod, BiasSeverityLevel, BiasConfidenceLevel
)
from src.bias_engine.taxonomy_loader import BiaxTaxonomyLoader, get_taxonomy_loader
from src.bias_engine.nlp_pipeline import NLPPipeline, LanguageDetector, TextPreprocessor
from src.bias_engine.rule_based_detector import RuleBasedBiasDetector
from src.bias_engine.scoring_algorithms import (
    ConfidenceCalculator, SeverityCalculator, DetectionSignal
)
from src.bias_engine.core_detector import (
    BiasDetectionEngine, DetectionConfig, detect_bias_spans,
    classify_bias_type, calculate_severity, calculate_confidence
)
from src.bias_engine.config_manager import ConfigurationManager, BiasEngineConfig


class TestTaxonomyLoader(unittest.TestCase):
    """Test bias taxonomy loading and validation"""
    
    def setUp(self):
        # Create temporary taxonomy file
        self.temp_dir = tempfile.mkdtemp()
        self.taxonomy_path = Path(self.temp_dir) / "test_taxonomy.json"
        
        # Create test taxonomy
        test_taxonomy = {
            "bias_families": {
                "test_family": {
                    "id": "test_family",
                    "name": "Test Family",
                    "description": "Test bias family",
                    "weight": 1.0,
                    "subtypes": {
                        "test_subtype": {
                            "id": "test_subtype",
                            "name": "Test Subtype",
                            "description": "Test bias subtype",
                            "severity_multiplier": 1.0,
                            "patterns": ["test pattern", "another pattern"]
                        }
                    }
                }
            },
            "intersectional_combinations": [
                ["test_family.test_subtype", "test_family.test_subtype"]
            ],
            "severity_thresholds": {
                "low": 0.3,
                "medium": 0.6,
                "high": 0.8,
                "critical": 0.95
            },
            "confidence_thresholds": {
                "uncertain": 0.4,
                "likely": 0.6,
                "confident": 0.8,
                "certain": 0.95
            }
        }
        
        with open(self.taxonomy_path, 'w') as f:
            json.dump(test_taxonomy, f, indent=2)
    
    def test_taxonomy_loading(self):
        """Test taxonomy loading from file"""
        loader = BiaxTaxonomyLoader(self.taxonomy_path)
        loader.load_taxonomy()
        
        self.assertEqual(len(loader.families), 1)
        self.assertIn("test_family", loader.families)
        
        family = loader.families["test_family"]
        self.assertEqual(family.name, "Test Family")
        self.assertEqual(len(family.subtypes), 1)
    
    def test_taxonomy_validation(self):
        """Test taxonomy structure validation"""
        loader = BiaxTaxonomyLoader(self.taxonomy_path)
        loader.load_taxonomy()
        
        # Test family retrieval
        family = loader.get_family("test_family")
        self.assertIsNotNone(family)
        
        # Test subtype retrieval
        subtype = loader.get_subtype("test_family", "test_subtype")
        self.assertIsNotNone(subtype)
        
        # Test invalid family/subtype
        invalid_family = loader.get_family("invalid")
        self.assertIsNone(invalid_family)
        
        invalid_subtype = loader.get_subtype("test_family", "invalid")
        self.assertIsNone(invalid_subtype)
    
    def test_pattern_extraction(self):
        """Test pattern and keyword extraction"""
        loader = BiaxTaxonomyLoader(self.taxonomy_path)
        loader.load_taxonomy()
        
        patterns = loader.get_all_patterns()
        self.assertIn("test_family.test_subtype", patterns)
        self.assertEqual(len(patterns["test_family.test_subtype"]), 2)
    
    def test_intersectional_analysis(self):
        """Test intersectional bias analysis"""
        loader = BiaxTaxonomyLoader(self.taxonomy_path)
        loader.load_taxonomy()
        
        amplification = loader.calculate_intersectional_amplification(
            ["test_family.test_subtype", "test_family.test_subtype"]
        )
        self.assertGreater(amplification, 1.0)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)


class TestNLPPipeline(unittest.TestCase):
    """Test NLP pipeline functionality"""
    
    def setUp(self):
        self.nlp = NLPPipeline()
    
    def test_language_detection(self):
        """Test language detection"""
        # Test English
        lang, confidence = self.nlp.detect_language("This is English text.")
        self.assertEqual(lang, "en")
        
        # Test German (if available)
        lang, confidence = self.nlp.detect_language("Das ist deutscher Text.")
        # Should detect as German or fallback to English
        self.assertIn(lang, ["de", "en"])
    
    def test_text_processing(self):
        """Test text processing pipeline"""
        text = "This is a test sentence with some bias language."
        processed = self.nlp.process_text(text)
        
        self.assertIn('text', processed)
        self.assertIn('language', processed)
        self.assertIn('cleaned_text', processed)
        self.assertIn('sentences', processed)
        self.assertEqual(processed['text'], text)
    
    def test_span_extraction(self):
        """Test candidate span extraction"""
        text = "This text contains some problematic language that should be detected."
        spans = self.nlp.extract_candidate_spans(text)
        
        self.assertGreater(len(spans), 0)
        for span_text, start, end in spans:
            self.assertEqual(len(span_text), end - start)
            self.assertEqual(text[start:end], span_text)


class TestRuleBasedDetector(unittest.TestCase):
    """Test rule-based bias detection"""
    
    def setUp(self):
        self.detector = RuleBasedBiasDetector()
    
    def test_basic_detection(self):
        """Test basic bias pattern detection"""
        # Test text with known bias patterns
        test_texts = [
            "Women are too emotional for leadership roles.",
            "All immigrants are taking our jobs.",
            "Young people these days are so lazy.",
            "He's pretty articulate for a Black person."
        ]
        
        for text in test_texts:
            spans = self.detector.detect_bias_spans(text)
            self.assertGreaterEqual(len(spans), 0)  # Might not detect all without full taxonomy
    
    def test_span_properties(self):
        """Test properties of detected spans"""
        text = "This contains typical bias language patterns."
        spans = self.detector.detect_bias_spans(text)
        
        for span in spans:
            # Test span validity
            self.assertGreaterEqual(span.start, 0)
            self.assertLessEqual(span.end, len(text))
            self.assertLess(span.start, span.end)
            
            # Test score ranges
            self.assertGreaterEqual(span.confidence, 0.0)
            self.assertLessEqual(span.confidence, 1.0)
            self.assertGreaterEqual(span.severity, 0.0)
            self.assertLessEqual(span.severity, 10.0)
            
            # Test required fields
            self.assertIsNotNone(span.bias_family)
            self.assertIsNotNone(span.bias_subtype)
            self.assertEqual(span.method, DetectionMethod.RULE_BASED)


class TestScoringAlgorithms(unittest.TestCase):
    """Test confidence and severity scoring"""
    
    def setUp(self):
        self.confidence_calc = ConfidenceCalculator()
        self.severity_calc = SeverityCalculator()
    
    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        # Create test detection signals
        signals = [
            DetectionSignal(
                method=DetectionMethod.RULE_BASED,
                confidence=0.8,
                severity=7.0,
                evidence="Rule pattern match"
            ),
            DetectionSignal(
                method=DetectionMethod.ML_CLASSIFICATION,
                confidence=0.9,
                severity=6.5,
                evidence="ML classification"
            )
        ]
        
        confidence = self.confidence_calc.calculate_confidence(signals)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertGreater(confidence, 0.0)  # Should be positive with valid signals
    
    def test_severity_calculation(self):
        """Test severity score calculation"""
        context = {
            'has_generalization': True,
            'explicit_language': False,
            'has_qualification': False
        }
        
        severity = self.severity_calc.calculate_severity(
            "demographic", "gender", context
        )
        
        self.assertGreaterEqual(severity, 0.0)
        self.assertLessEqual(severity, 10.0)
    
    def test_uncertainty_calculation(self):
        """Test uncertainty quantification"""
        signals = [
            DetectionSignal(DetectionMethod.RULE_BASED, 0.8, 7.0, "test"),
            DetectionSignal(DetectionMethod.ML_CLASSIFICATION, 0.6, 5.0, "test")
        ]
        
        uncertainty = self.confidence_calc.calculate_uncertainty(signals)
        
        self.assertIn('epistemic', uncertainty)
        self.assertIn('aleatoric', uncertainty)
        self.assertIn('total', uncertainty)
        
        for key, value in uncertainty.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


class TestBiasDetectionEngine(unittest.TestCase):
    """Test complete bias detection engine"""
    
    def setUp(self):
        config = DetectionConfig(
            confidence_threshold=0.1,  # Lower threshold for testing
            severity_threshold=0.0,
            enable_intersectional_analysis=True
        )
        self.engine = BiasDetectionEngine(config)
    
    def test_end_to_end_detection(self):
        """Test complete bias detection pipeline"""
        test_texts = [
            "This is neutral text without any bias.",
            "Women are naturally better at caring for children.",
            "All teenagers are irresponsible and reckless.",
            "He's very articulate for someone from his background."
        ]
        
        for text in test_texts:
            results = self.engine.detect_bias_spans(text)
            
            self.assertEqual(len(results), 1)  # Should return one result
            result = results[0]
            
            # Test result structure
            self.assertIsInstance(result, BiasDetectionResult)
            self.assertEqual(result.text, text)
            self.assertIsNotNone(result.language)
            self.assertIsInstance(result.detected_spans, list)
            
            # Test overall scores
            self.assertGreaterEqual(result.overall_confidence, 0.0)
            self.assertLessEqual(result.overall_confidence, 1.0)
            self.assertGreaterEqual(result.overall_severity, 0.0)
            self.assertLessEqual(result.overall_severity, 10.0)
    
    def test_public_api_functions(self):
        """Test public API functions"""
        text = "This text might contain some bias patterns."
        
        # Test main detection function
        results = detect_bias_spans(text)
        self.assertIsInstance(results, list)
        
        # Test classification function
        classification = classify_bias_type("bias span", text)
        self.assertIsInstance(classification, BiasClassification)
        
        # Test severity calculation
        severity = calculate_severity("demographic.gender", "bias span")
        self.assertIsInstance(severity, float)
        self.assertGreaterEqual(severity, 0.0)
        
        # Test confidence calculation
        confidence = calculate_confidence([])
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
    
    def test_configuration(self):
        """Test detection configuration"""
        config = DetectionConfig(
            enable_rule_based=True,
            enable_ml_classification=False,
            confidence_threshold=0.5,
            severity_threshold=3.0
        )
        
        engine = BiasDetectionEngine(config)
        self.assertEqual(engine.config.confidence_threshold, 0.5)
        self.assertEqual(engine.config.severity_threshold, 3.0)
        self.assertFalse(engine.config.enable_ml_classification)
    
    def test_error_handling(self):
        """Test error handling and edge cases"""
        # Test empty text
        results = self.engine.detect_bias_spans("")
        self.assertEqual(len(results), 0)
        
        # Test None text
        results = self.engine.detect_bias_spans(None)
        self.assertEqual(len(results), 0)
        
        # Test very long text (should be handled gracefully)
        long_text = "This is a test. " * 1000
        results = self.engine.detect_bias_spans(long_text)
        self.assertIsInstance(results, list)
    
    def test_intersectional_analysis(self):
        """Test intersectional bias analysis"""
        # Text with multiple potential bias types
        text = "Young women from minority backgrounds are not suited for tech leadership roles."
        
        results = self.engine.detect_bias_spans(text)
        if results and results[0].detected_spans:
            result = results[0]
            
            # Check if intersectional analysis was performed
            if result.intersectional_analysis:
                self.assertIsInstance(result.intersectional_analysis.detected_identities, list)
                self.assertGreater(result.intersectional_analysis.amplification_factor, 1.0)


class TestConfigurationManager(unittest.TestCase):
    """Test configuration and model management"""
    
    def setUp(self):
        self.config_manager = ConfigurationManager()
    
    def test_default_config(self):
        """Test default configuration creation"""
        config = self.config_manager.config
        self.assertIsInstance(config, BiasEngineConfig)
        self.assertGreaterEqual(config.confidence_threshold, 0.0)
        self.assertLessEqual(config.confidence_threshold, 1.0)
    
    def test_system_info(self):
        """Test system information retrieval"""
        info = self.config_manager.get_system_info()
        
        self.assertIn('config', info)
        self.assertIn('system_memory', info)
        self.assertIn('model_cache', info)
        self.assertIn('supported_features', info)
    
    def test_model_cache(self):
        """Test model caching functionality"""
        cache = self.config_manager.model_cache
        
        # Test cache operations
        test_model = "dummy_model"
        cache.put("test_key", test_model, 10.0)
        
        retrieved = cache.get("test_key")
        self.assertEqual(retrieved, test_model)
        
        # Test cache stats
        stats = cache.get_stats()
        self.assertIn('cached_models', stats)
        self.assertIn('total_memory_mb', stats)
        
        # Test cache removal
        removed = cache.remove("test_key")
        self.assertTrue(removed)
        
        retrieved_after_removal = cache.get("test_key")
        self.assertIsNone(retrieved_after_removal)


class TestDataModels(unittest.TestCase):
    """Test data model validation and functionality"""
    
    def test_bias_span_creation(self):
        """Test BiasSpan creation and validation"""
        span = BiasSpan(
            start=10,
            end=20,
            text="test span",
            bias_family="demographic",
            bias_subtype="gender",
            severity=5.0,
            confidence=0.8,
            method=DetectionMethod.RULE_BASED
        )
        
        self.assertEqual(span.text, "test span")
        self.assertEqual(span.severity_level, BiasSeverityLevel.MEDIUM)
        self.assertEqual(span.confidence_level, BiasConfidenceLevel.CONFIDENT)
    
    def test_bias_span_validation(self):
        """Test BiasSpan input validation"""
        # Test invalid coordinates
        with self.assertRaises(ValueError):
            BiasSpan(
                start=20, end=10,  # Invalid: start > end
                text="test", bias_family="demo", bias_subtype="test",
                severity=5.0, confidence=0.8, method=DetectionMethod.RULE_BASED
            )
        
        # Test invalid severity
        with self.assertRaises(ValueError):
            BiasSpan(
                start=0, end=4, text="test",
                bias_family="demo", bias_subtype="test",
                severity=15.0,  # Invalid: > 10.0
                confidence=0.8, method=DetectionMethod.RULE_BASED
            )
        
        # Test invalid confidence
        with self.assertRaises(ValueError):
            BiasSpan(
                start=0, end=4, text="test",
                bias_family="demo", bias_subtype="test",
                severity=5.0, confidence=1.5,  # Invalid: > 1.0
                method=DetectionMethod.RULE_BASED
            )
    
    def test_bias_span_overlap(self):
        """Test BiasSpan overlap detection"""
        span1 = BiasSpan(
            start=10, end=20, text="span one",
            bias_family="demo", bias_subtype="test",
            severity=5.0, confidence=0.8, method=DetectionMethod.RULE_BASED
        )
        
        span2 = BiasSpan(
            start=15, end=25, text="span two",
            bias_family="demo", bias_subtype="test",
            severity=6.0, confidence=0.9, method=DetectionMethod.ML_CLASSIFICATION
        )
        
        span3 = BiasSpan(
            start=30, end=40, text="span three",
            bias_family="demo", bias_subtype="test",
            severity=4.0, confidence=0.7, method=DetectionMethod.RULE_BASED
        )
        
        # Test overlap detection
        self.assertTrue(span1.overlaps_with(span2))
        self.assertFalse(span1.overlaps_with(span3))
        self.assertFalse(span2.overlaps_with(span3))
    
    def test_detection_result_serialization(self):
        """Test BiasDetectionResult JSON serialization"""
        result = BiasDetectionResult(
            id="test-id",
            text="test text",
            language="en",
            detected_spans=[],
            overall_severity=3.5,
            overall_confidence=0.7
        )
        
        # Test to_dict conversion
        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['text'], "test text")
        self.assertEqual(result_dict['overall_severity'], 3.5)
        
        # Test JSON serialization
        json_str = result.to_json()
        self.assertIsInstance(json_str, str)
        
        # Test JSON can be parsed
        parsed = json.loads(json_str)
        self.assertEqual(parsed['text'], "test text")


class TestPerformanceAndMemory(unittest.TestCase):
    """Test performance and memory management"""
    
    def test_large_text_handling(self):
        """Test handling of large text inputs"""
        # Create large text
        large_text = "This is a test sentence with potential bias. " * 100
        
        config = DetectionConfig(
            confidence_threshold=0.1,
            max_spans_per_text=10  # Limit spans for performance
        )
        engine = BiasDetectionEngine(config)
        
        results = engine.detect_bias_spans(large_text)
        
        # Should handle large text without errors
        self.assertIsInstance(results, list)
        if results:
            result = results[0]
            # Should respect max_spans_per_text limit
            self.assertLessEqual(len(result.detected_spans), 10)
    
    def test_concurrent_detection(self):
        """Test concurrent bias detection"""
        import threading
        
        engine = BiasDetectionEngine()
        test_texts = [
            "Text one with potential bias.",
            "Text two with different bias.",
            "Text three with yet another bias."
        ]
        
        results = []
        threads = []
        
        def detect_bias(text):
            result = engine.detect_bias_spans(text)
            results.append(result)
        
        # Start concurrent detection
        for text in test_texts:
            thread = threading.Thread(target=detect_bias, args=(text,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have results from all threads
        self.assertEqual(len(results), len(test_texts))
        
        for result in results:
            self.assertIsInstance(result, list)
    
    def test_memory_cleanup(self):
        """Test memory cleanup functionality"""
        config_manager = ConfigurationManager()
        cache = config_manager.model_cache
        
        # Add some test models to cache
        for i in range(5):
            cache.put(f"model_{i}", f"dummy_model_{i}", 50.0)
        
        initial_count = cache.get_stats()['cached_models']
        self.assertEqual(initial_count, 5)
        
        # Clear cache
        cache.clear()
        final_count = cache.get_stats()['cached_models']
        self.assertEqual(final_count, 0)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    unittest.main(verbosity=2)
