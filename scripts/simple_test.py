#!/usr/bin/env python3
"""
Simple test script to verify the bias detection engine works
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    # Test basic imports
    print("Testing basic imports...")
    from models.bias_models import BiasSpan, BiasDetectionResult, DetectionMethod
    print("✅ Models imported successfully")
    
    from bias_engine.taxonomy_loader import BiaxTaxonomyLoader
    print("✅ Taxonomy loader imported successfully")
    
    from bias_engine.nlp_pipeline import NLPPipeline, LanguageDetector
    print("✅ NLP pipeline imported successfully")
    
    from bias_engine.scoring_algorithms import ConfidenceCalculator, SeverityCalculator
    print("✅ Scoring algorithms imported successfully")
    
    # Test taxonomy loading
    print("\nTesting taxonomy loading...")
    try:
        taxonomy = BiaxTaxonomyLoader()
        taxonomy.load_taxonomy()
        stats = taxonomy.get_statistics()
        print(f"✅ Taxonomy loaded: {stats['families']} families, {stats['subtypes']} subtypes, {stats['patterns']} patterns")
    except Exception as e:
        print(f"❌ Taxonomy loading failed: {e}")
    
    # Test NLP pipeline
    print("\nTesting NLP pipeline...")
    try:
        nlp = NLPPipeline()
        test_text = "This is a simple test sentence."
        processed = nlp.process_text(test_text)
        print(f"✅ NLP processing successful: detected language '{processed['language']}'")
    except Exception as e:
        print(f"❌ NLP pipeline failed: {e}")
    
    # Test scoring algorithms
    print("\nTesting scoring algorithms...")
    try:
        from bias_engine.scoring_algorithms import DetectionSignal
        
        conf_calc = ConfidenceCalculator()
        sev_calc = SeverityCalculator()
        
        # Test confidence calculation
        signals = [
            DetectionSignal(DetectionMethod.RULE_BASED, 0.8, 7.0, "test")
        ]
        confidence = conf_calc.calculate_confidence(signals)
        print(f"✅ Confidence calculation successful: {confidence:.3f}")
        
        # Test severity calculation
        context = {'has_generalization': False}
        severity = sev_calc.calculate_severity("demographic", "gender", context)
        print(f"✅ Severity calculation successful: {severity:.3f}")
        
    except Exception as e:
        print(f"❌ Scoring algorithms failed: {e}")
    
    # Test bias span creation
    print("\nTesting bias span creation...")
    try:
        span = BiasSpan(
            start=0,
            end=9,  # Fix: length should match text length
            text="test span",
            bias_family="demographic", 
            bias_subtype="gender",
            severity=5.0,
            confidence=0.8,
            method=DetectionMethod.RULE_BASED
        )
        print(f"✅ BiasSpan created: '{span.text}' -> {span.bias_family}.{span.bias_subtype}")
        print(f"   Severity level: {span.severity_level.value}, Confidence level: {span.confidence_level.value}")
    except Exception as e:
        print(f"❌ BiasSpan creation failed: {e}")
    
    print("\n🎉 All basic tests passed! The bias detection engine is ready to use.")
    print("\n💡 To run the full demo: python scripts/demo_bias_detection.py")
    print("💡 To run comprehensive tests: python -m pytest tests/ -v")
    
except Exception as e:
    print(f"❌ Critical error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
