#!/usr/bin/env python3
"""
Bias Detection Engine Demo

Demonstrates the capabilities of the bias detection engine with real examples.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bias_engine import (
    detect_bias_spans,
    get_detection_engine,
    DetectionConfig
)
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_separator(title: str) -> None:
    """Print a section separator"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def demo_basic_detection():
    """Demonstrate basic bias detection"""
    print_separator("BASIC BIAS DETECTION DEMO")
    
    test_texts = [
        "Women are naturally more emotional than men.",
        "All teenagers are irresponsible and can't be trusted.",
        "He's very articulate for someone from his background.",
        "Immigrants are taking jobs away from real Americans.",
        "This is neutral text without any bias.",
        "Older workers are less productive and adaptable to new technology."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Analyzing: \"{text}\"")
        print("-" * 60)
        
        try:
            results = detect_bias_spans(text)
            
            if not results or not results[0].detected_spans:
                print("✅ No bias detected")
            else:
                result = results[0]
                print(f"🔍 Overall Severity: {result.overall_severity:.2f}/10")
                print(f"🎯 Overall Confidence: {result.overall_confidence:.2f}")
                print(f"📊 Language: {result.language}")
                
                for j, span in enumerate(result.detected_spans, 1):
                    print(f"\n   Bias Span {j}:")
                    print(f"   📍 Text: \"{span.text}\"")
                    print(f"   📈 Type: {span.bias_family}.{span.bias_subtype}")
                    print(f"   ⚠️  Severity: {span.severity:.2f}/10 ({span.severity_level.value})")
                    print(f"   🎯 Confidence: {span.confidence:.2f} ({span.confidence_level.value})")
                    print(f"   🔧 Method: {span.method.value}")
                    if span.explanation:
                        print(f"   💡 Explanation: {span.explanation}")
                
                if result.intersectional_analysis:
                    ia = result.intersectional_analysis
                    print(f"\n   🔗 Intersectional Analysis:")
                    print(f"   👥 Identities: {', '.join(ia.detected_identities)}")
                    print(f"   📈 Amplification Factor: {ia.amplification_factor:.2f}")
                    if ia.marginalization_indicators:
                        print(f"   ⚠️  Marginalization: {', '.join(ia.marginalization_indicators)}")
                    if ia.erasure_indicators:
                        print(f"   🚫 Erasure: {', '.join(ia.erasure_indicators)}")
        
        except Exception as e:
            print(f"❌ Error analyzing text: {e}")
        
        print("\n")


def demo_advanced_configuration():
    """Demonstrate advanced configuration options"""
    print_separator("ADVANCED CONFIGURATION DEMO")
    
    # Create custom configuration
    config = DetectionConfig(
        enable_rule_based=True,
        enable_ml_classification=True,
        enable_intersectional_analysis=True,
        confidence_threshold=0.2,  # Lower threshold for demo
        severity_threshold=1.0,
        max_spans_per_text=20,
        include_low_confidence=True
    )
    
    engine = get_detection_engine(config)
    
    print("Configuration:")
    print(f"  • Rule-based detection: {config.enable_rule_based}")
    print(f"  • ML classification: {config.enable_ml_classification}")
    print(f"  • Intersectional analysis: {config.enable_intersectional_analysis}")
    print(f"  • Confidence threshold: {config.confidence_threshold}")
    print(f"  • Severity threshold: {config.severity_threshold}")
    print(f"  • Max spans per text: {config.max_spans_per_text}")
    
    # Complex text with multiple bias types
    complex_text = (
        "Young women from minority backgrounds are typically not suited for "
        "technical leadership roles because they lack the natural logical "
        "thinking abilities that men possess. Additionally, their emotional "
        "nature makes them unreliable under pressure, especially when dealing "
        "with foreign clients who expect professional behavior."
    )
    
    print(f"\n\nAnalyzing complex text:")
    print(f"Text: \"{complex_text}\"")
    print("-" * 80)
    
    try:
        results = engine.detect_bias_spans(complex_text)
        
        if results:
            result = results[0]
            
            print(f"\n🔍 Detection Results:")
            print(f"   Overall Severity: {result.overall_severity:.2f}/10")
            print(f"   Overall Confidence: {result.overall_confidence:.2f}")
            print(f"   Total Spans Detected: {len(result.detected_spans)}")
            print(f"   Bias Families: {', '.join(result.bias_families_detected)}")
            
            # Group spans by bias family
            family_spans = {}
            for span in result.detected_spans:
                if span.bias_family not in family_spans:
                    family_spans[span.bias_family] = []
                family_spans[span.bias_family].append(span)
            
            print(f"\n📊 Bias Analysis by Family:")
            for family, spans in family_spans.items():
                avg_severity = sum(s.severity for s in spans) / len(spans)
                avg_confidence = sum(s.confidence for s in spans) / len(spans)
                print(f"   {family}: {len(spans)} span(s), "
                      f"avg severity {avg_severity:.2f}, "
                      f"avg confidence {avg_confidence:.2f}")
            
            if result.intersectional_analysis:
                ia = result.intersectional_analysis
                print(f"\n🔗 Intersectional Analysis:")
                print(f"   Detected Identities: {ia.detected_identities}")
                print(f"   Intersection Score: {ia.intersection_score:.2f}")
                print(f"   Amplification Factor: {ia.amplification_factor:.2f}")
                
                if ia.marginalization_indicators:
                    print(f"   Marginalization Indicators: {ia.marginalization_indicators}")
                if ia.privilege_indicators:
                    print(f"   Privilege Indicators: {ia.privilege_indicators}")
                if ia.erasure_indicators:
                    print(f"   Erasure Indicators: {ia.erasure_indicators}")
    
    except Exception as e:
        print(f"❌ Error in advanced analysis: {e}")
        import traceback
        traceback.print_exc()


def demo_json_output():
    """Demonstrate JSON output format"""
    print_separator("JSON OUTPUT DEMO")
    
    text = "Typical millennial behavior - always expecting participation trophies and safe spaces."
    
    print(f"Analyzing: \"{text}\"")
    print("\nJSON Output:")
    print("-" * 40)
    
    try:
        results = detect_bias_spans(text)
        
        if results:
            result = results[0]
            json_output = result.to_json(indent=2)
            print(json_output)
        else:
            print("No results to display")
    
    except Exception as e:
        print(f"❌ Error generating JSON: {e}")


def demo_multilingual_support():
    """Demonstrate multilingual bias detection"""
    print_separator("MULTILINGUAL SUPPORT DEMO")
    
    multilingual_texts = [
        ("en", "Women drivers are naturally worse than men at parking."),
        ("de", "Ausländer nehmen uns die Arbeitsplätze weg."),  # Foreigners take away our jobs
        ("en", "All Muslims are potentially dangerous terrorists."),
        ("auto", "Les jeunes d'aujourd'hui sont paresseux.")  # Today's youth are lazy (French)
    ]
    
    for i, (lang, text) in enumerate(multilingual_texts, 1):
        print(f"\n{i}. Language: {lang}")
        print(f"   Text: \"{text}\"")
        print("   " + "-" * 50)
        
        try:
            results = detect_bias_spans(text, language=lang)
            
            if results and results[0].detected_spans:
                result = results[0]
                print(f"   ✅ Detected Language: {result.language}")
                print(f"   🔍 Spans Found: {len(result.detected_spans)}")
                print(f"   📊 Overall Severity: {result.overall_severity:.2f}/10")
                
                for span in result.detected_spans[:2]:  # Show first 2 spans
                    print(f"      • \"{span.text}\" → {span.bias_family}.{span.bias_subtype} "
                          f"(severity: {span.severity:.2f}, confidence: {span.confidence:.2f})")
            else:
                detected_lang = results[0].language if results else "unknown"
                print(f"   ✅ Detected Language: {detected_lang}")
                print(f"   ✅ No bias detected")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")


def demo_performance_metrics():
    """Demonstrate performance monitoring"""
    print_separator("PERFORMANCE METRICS DEMO")
    
    import time
    from bias_engine.config_manager import get_config_manager
    
    config_manager = get_config_manager()
    
    print("System Information:")
    system_info = config_manager.get_system_info()
    
    print(f"  💻 System Memory: {system_info['system_memory']['total_gb']:.1f}GB total, "
          f"{system_info['system_memory']['available_gb']:.1f}GB available "
          f"({system_info['system_memory']['usage_pct']:.1f}% used)")
    
    print(f"  🧠 Model Cache: {system_info['model_cache']['cached_models']} models, "
          f"{system_info['model_cache']['total_memory_mb']:.1f}MB used")
    
    print(f"  🔧 Supported Features:")
    for feature, available in system_info['supported_features'].items():
        status = "✅" if available else "❌"
        print(f"     {status} {feature}")
    
    # Performance test
    print("\n⏱️  Performance Test:")
    test_texts = [
        "This text contains potential bias against certain groups.",
        "Women are not suited for technical roles.",
        "All immigrants are criminals.",
        "Young people these days have no work ethic.",
        "Older workers should retire to make room for younger talent."
    ]
    
    start_time = time.time()
    total_spans = 0
    
    for text in test_texts:
        try:
            results = detect_bias_spans(text)
            if results:
                total_spans += len(results[0].detected_spans)
        except Exception as e:
            print(f"     Error processing text: {e}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"     📊 Processed {len(test_texts)} texts in {processing_time:.3f}s")
    print(f"     🎯 Average time per text: {processing_time/len(test_texts):.3f}s")
    print(f"     📈 Total bias spans detected: {total_spans}")
    print(f"     ⚡ Throughput: {len(test_texts)/processing_time:.1f} texts/second")


def main():
    """Run all demos"""
    print("🚀 Bias Detection Engine Demo")
    print("   Advanced intersectional bias detection with ML and rule-based methods")
    
    try:
        demo_basic_detection()
        demo_advanced_configuration()
        demo_json_output()
        demo_multilingual_support()
        demo_performance_metrics()
        
        print_separator("DEMO COMPLETED")
        print("✅ All demonstrations completed successfully!")
        print("\n📖 For more information, see the documentation and test suite.")
        print("🔧 To run tests: python -m pytest tests/ -v")
        print("📊 To see detailed configuration: check config/bias_families.json")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
