"""
Cultural Adaptation Engine Examples

This module demonstrates various use cases and integration patterns
for the Cultural Adaptation Engine with real-world scenarios.
"""

import json
from typing import Dict, Any
from bias_engine.cultural import (
    enhance_bias_with_culture,
    get_cultural_context,
    CulturalIntegration,
    cultural_bias_enhancer,
    with_cultural_context
)


# Example 1: Basic Cultural Enhancement
def basic_enhancement_example():
    """
    Basic example of enhancing bias detection with cultural context.
    """
    print("=== Basic Cultural Enhancement Example ===")

    # Simulated bias detection results
    bias_results = {
        "overall_bias_score": 0.72,
        "biases_detected": {
            "gender": {
                "severity": 0.8,
                "confidence": 0.91,
                "indicators": ["gendered language", "role assumptions"]
            },
            "cultural": {
                "severity": 0.65,
                "confidence": 0.87,
                "indicators": ["cultural stereotypes"]
            }
        },
        "text_analysis": {
            "total_words": 150,
            "flagged_phrases": 3
        }
    }

    # Enhance with cultural context (German to Japanese communication)
    enhanced_results = enhance_bias_with_culture(
        bias_results=bias_results,
        sender_culture="DE",
        receiver_culture="JP",
        context={"type": "business", "urgency": "medium"}
    )

    print("Original gender bias severity:", bias_results["biases_detected"]["gender"]["severity"])
    print("Enhanced gender bias severity:", enhanced_results["biases_detected"]["gender"]["severity"])
    print("Cultural modifier applied:", enhanced_results["cultural_adjustments"]["gender"]["cultural_modifier"])
    print("Cultural distance:", enhanced_results["cultural_context"]["cultural_distance"])
    print("Communication risk level:", enhanced_results["cultural_communication_risk"]["level"])

    return enhanced_results


# Example 2: Educational Content Analysis
def educational_content_example():
    """
    Example of analyzing educational content for cultural bias.
    """
    print("\n=== Educational Content Analysis Example ===")

    # Simulated analysis of course material
    bias_results = {
        "overall_bias_score": 0.58,
        "biases_detected": {
            "cultural": {
                "severity": 0.7,
                "confidence": 0.85,
                "indicators": ["western-centric examples", "cultural assumptions"]
            },
            "socioeconomic": {
                "severity": 0.45,
                "confidence": 0.78,
                "indicators": ["affluence assumptions"]
            }
        }
    }

    # Analyze US instructor creating content for Chinese students
    enhanced_results = enhance_bias_with_culture(
        bias_results=bias_results,
        sender_culture="US",
        receiver_culture="CN",
        context={
            "type": "educational",
            "level": "university",
            "subject": "business_studies"
        }
    )

    # Print cultural analysis insights
    print("Cross-cultural Analysis:")
    analysis = enhanced_results["cross_cultural_analysis"]
    print(f"  Cultural Distance: {analysis['cultural_distance']:.1f}")
    print(f"  Bridge Score: {analysis['bridge_score']:.1f}")
    print(f"  Overall Risk: {analysis['overall_risk']}")

    print("\nCultural Intelligence Insights:")
    intelligence = enhanced_results["cultural_intelligence"]
    if intelligence["sensitivity_warnings"]:
        for warning in intelligence["sensitivity_warnings"]:
            print(f"  Warning: {warning['message']}")

    return enhanced_results


# Example 3: Business Communication Dashboard
def business_communication_dashboard():
    """
    Example of generating dashboard data for business communication analysis.
    """
    print("\n=== Business Communication Dashboard Example ===")

    # Initialize cultural integration
    integration = CulturalIntegration(enable_caching=True)

    # Generate dashboard data for multinational team
    cultures = ["DE", "US", "JP", "CN", "FR"]
    dashboard_data = integration.get_cultural_dashboard_data(cultures)

    print("Dashboard Data Generated:")
    print(f"  Cultures analyzed: {len(dashboard_data['cultures'])}")
    print(f"  Cultural distances calculated: {len(dashboard_data['cultural_distances'])}")
    print(f"  Communication styles analyzed: {len(dashboard_data['communication_styles'])}")

    # Show some sample data
    print("\nSample Cultural Distances:")
    for pair, distance in list(dashboard_data["cultural_distances"].items())[:3]:
        print(f"  {pair}: {distance:.1f}")

    print("\nSample Communication Styles:")
    for culture, style in list(dashboard_data["communication_styles"].items())[:2]:
        print(f"  {culture}: {style['directness']} directness, {style['formality']} formality")

    return dashboard_data


# Example 4: Healthcare Communication
def healthcare_communication_example():
    """
    Example of analyzing healthcare provider-patient communication.
    """
    print("\n=== Healthcare Communication Example ===")

    # Simulated bias analysis of doctor-patient interaction
    bias_results = {
        "overall_bias_score": 0.43,
        "biases_detected": {
            "age": {
                "severity": 0.6,
                "confidence": 0.82,
                "indicators": ["age-related assumptions"]
            },
            "cultural": {
                "severity": 0.35,
                "confidence": 0.75,
                "indicators": ["cultural health beliefs"]
            }
        }
    }

    # German doctor communicating with Spanish patient
    enhanced_results = enhance_bias_with_culture(
        bias_results=bias_results,
        sender_culture="DE",
        receiver_culture="ES",
        context={
            "type": "healthcare",
            "sensitive": True,
            "patient_age": "elderly"
        }
    )

    # Generate specific recommendations
    cultural_context = get_cultural_context("DE", "ES", {"type": "healthcare"})

    print("Healthcare Communication Analysis:")
    print(f"  Adjusted age bias severity: {enhanced_results['biases_detected']['age']['severity']:.2f}")
    print(f"  Cultural risk level: {enhanced_results['cultural_communication_risk']['level']}")

    print("\nCommunication Recommendations:")
    strategies = cultural_context["communication_strategies"]
    for adaptation in strategies.get("sender_adaptations", [])[:2]:
        print(f"  • {adaptation}")

    return enhanced_results


# Example 5: Decorator-based Integration
@cultural_bias_enhancer
def analyze_email_bias(email_text: str, sender_culture=None, receiver_culture=None, context=None) -> Dict[str, Any]:
    """
    Example function using the cultural bias enhancer decorator.
    """
    # Simulated email bias analysis
    bias_score = len([word for word in email_text.lower().split()
                     if word in ["guys", "ladies", "oriental", "exotic"]]) * 0.2

    return {
        "overall_bias_score": min(bias_score, 1.0),
        "biases_detected": {
            "gender": {"severity": min(bias_score * 1.2, 1.0), "confidence": 0.8},
            "cultural": {"severity": min(bias_score * 0.8, 1.0), "confidence": 0.7}
        },
        "email_metadata": {
            "word_count": len(email_text.split()),
            "flagged_terms": min(int(bias_score * 5), 5)
        }
    }


def decorator_integration_example():
    """
    Example using decorator-based cultural integration.
    """
    print("\n=== Decorator Integration Example ===")

    # Sample email text
    email_text = "Hi guys, I need your help with the exotic new project from our oriental partners."

    # Analyze with automatic cultural enhancement
    results = analyze_email_bias(
        email_text,
        sender_culture="US",
        receiver_culture="JP",
        context={"type": "business", "urgency": "medium"}
    )

    print(f"Email analysis completed with cultural enhancement")
    if "cultural_context" in results:
        print(f"  Cultural distance: {results['cultural_context']['cultural_distance']:.1f}")
        print(f"  Communication risk: {results['cultural_communication_risk']['level']}")

    print(f"  Overall bias score: {results['overall_bias_score']:.2f}")

    return results


# Example 6: Custom Cultural Analysis
def custom_cultural_analysis():
    """
    Example of custom cultural analysis with hooks and customization.
    """
    print("\n=== Custom Cultural Analysis Example ===")

    # Initialize integration with custom hooks
    integration = CulturalIntegration(enable_caching=True)

    # Define custom hooks
    def audit_hook(bias_results, sender, receiver, context):
        print(f"    Audit: Analyzing {sender} → {receiver} communication")
        # Could log to audit system here

    def enrichment_hook(enhanced_results, sender, receiver, context):
        print(f"    Enrichment: Enhanced bias analysis completed")
        # Could add custom metrics here

    # Register hooks
    integration.register_pre_bias_hook(audit_hook)
    integration.register_post_bias_hook(enrichment_hook)

    # Sample bias results
    bias_results = {
        "overall_bias_score": 0.55,
        "biases_detected": {
            "political": {"severity": 0.6, "confidence": 0.85}
        }
    }

    # Enhanced analysis with hooks
    enhanced_results = integration.enhance_bias_detection(
        bias_results,
        sender_culture="FR",
        receiver_culture="DE",
        context={"type": "political_discussion"}
    )

    print("Custom analysis completed with hooks executed")
    print(f"  Political bias adjustment: {enhanced_results['cultural_adjustments']['political']['explanation']}")

    return enhanced_results


# Example 7: Multi-cultural Team Analysis
def multicultural_team_analysis():
    """
    Example of analyzing communication patterns in a multicultural team.
    """
    print("\n=== Multi-cultural Team Analysis Example ===")

    # Team composition
    team_cultures = ["US", "DE", "JP", "IT", "CN"]

    # Analyze all possible cultural pairs
    integration = CulturalIntegration()

    print("Team Cultural Analysis:")

    # Generate pairwise analysis
    for i, culture1 in enumerate(team_cultures):
        for culture2 in team_cultures[i+1:]:
            context_analysis = get_cultural_context(
                culture1, culture2,
                {"type": "business", "team_setting": True}
            )

            distance = context_analysis["cultural_distance"]
            risk = context_analysis["overall_risk"]

            print(f"  {culture1} ↔ {culture2}: Distance={distance:.1f}, Risk={risk}")

            # Highlight high-risk pairs
            if distance > 50:
                print(f"    ⚠️  High cultural distance - requires attention")

    # Generate team recommendations
    dashboard_data = integration.get_cultural_dashboard_data(team_cultures)

    print("\nTeam Communication Recommendations:")
    print("  • Use clear, explicit communication to bridge cultural gaps")
    print("  • Establish team norms that respect all cultural styles")
    print("  • Provide cultural awareness training for high-distance pairs")
    print("  • Create structured meeting formats to accommodate different styles")

    return dashboard_data


# Example 8: Real-time Cultural Adaptation
def realtime_adaptation_example():
    """
    Example of real-time cultural adaptation in a conversation system.
    """
    print("\n=== Real-time Cultural Adaptation Example ===")

    # Simulate a conversation with multiple exchanges
    conversation_history = []

    # Conversation between German and Chinese participants
    exchanges = [
        ("DE", "CN", "Let's discuss the project timeline directly."),
        ("CN", "DE", "We should consider all stakeholders carefully."),
        ("DE", "CN", "I need a concrete decision by tomorrow."),
        ("CN", "DE", "Perhaps we could schedule another meeting to discuss further.")
    ]

    integration = CulturalIntegration(enable_caching=True)

    for sender, receiver, message in exchanges:
        # Simulate bias detection
        bias_results = {
            "overall_bias_score": 0.3,
            "biases_detected": {
                "cultural": {"severity": 0.4, "confidence": 0.7}
            },
            "message": message
        }

        # Apply cultural adaptation
        enhanced_results = integration.enhance_bias_detection(
            bias_results, sender, receiver,
            context={"type": "business", "real_time": True}
        )

        # Store in conversation history
        conversation_history.append({
            "sender": sender,
            "receiver": receiver,
            "message": message,
            "cultural_risk": enhanced_results["cultural_communication_risk"]["level"],
            "adjusted_severity": enhanced_results["biases_detected"]["cultural"]["severity"]
        })

    print("Real-time Conversation Analysis:")
    for i, exchange in enumerate(conversation_history, 1):
        print(f"  Exchange {i}: {exchange['sender']} → {exchange['receiver']}")
        print(f"    Risk: {exchange['cultural_risk']}")
        print(f"    Message: \"{exchange['message'][:50]}...\"")

    return conversation_history


# Example 9: A/B Testing Cultural Variants
@with_cultural_context()
def analyze_content_variants(content_a: str, content_b: str,
                           sender_culture=None, receiver_culture=None, context=None):
    """
    Example function using cultural context decorator for A/B testing.
    """
    # Simulate bias analysis for two content variants
    def analyze_variant(content):
        bias_indicators = ["assumption", "stereotype", "exclusive"]
        score = sum(1 for indicator in bias_indicators if indicator in content.lower()) * 0.2
        return {
            "bias_score": min(score, 1.0),
            "flagged_terms": [term for term in bias_indicators if term in content.lower()]
        }

    variant_a = analyze_variant(content_a)
    variant_b = analyze_variant(content_b)

    return {
        "variant_a": variant_a,
        "variant_b": variant_b,
        "recommendation": "variant_a" if variant_a["bias_score"] < variant_b["bias_score"] else "variant_b"
    }


def ab_testing_example():
    """
    Example of A/B testing content variants with cultural adaptation.
    """
    print("\n=== A/B Testing with Cultural Context Example ===")

    content_a = "Our team members should leverage their diverse backgrounds and unique assumptions."
    content_b = "Our team members should utilize their varied experiences and different perspectives."

    # Test with cultural context
    results = analyze_content_variants(
        content_a, content_b,
        sender_culture="US",
        receiver_culture="JP",
        context={"type": "business", "testing": True}
    )

    print("A/B Testing Results:")
    print(f"  Variant A bias score: {results['variant_a']['bias_score']:.2f}")
    print(f"  Variant B bias score: {results['variant_b']['bias_score']:.2f}")
    print(f"  Recommended variant: {results['recommendation']}")

    if "cultural_context" in results:
        print(f"  Cultural distance considered: {results['cultural_context']['cultural_distance']:.1f}")

    return results


# Main execution
def main():
    """
    Run all examples to demonstrate the Cultural Adaptation Engine.
    """
    print("Cultural Adaptation Engine - Comprehensive Examples\n")

    try:
        # Run all examples
        basic_enhancement_example()
        educational_content_example()
        business_communication_dashboard()
        healthcare_communication_example()
        decorator_integration_example()
        custom_cultural_analysis()
        multicultural_team_analysis()
        realtime_adaptation_example()
        ab_testing_example()

        print("\n=== All Examples Completed Successfully ===")
        print("The Cultural Adaptation Engine is ready for integration!")

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Please ensure the bias_engine.cultural module is properly installed.")


if __name__ == "__main__":
    main()