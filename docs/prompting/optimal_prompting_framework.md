# Optimal LLM Prompting Framework for Bias Detection & Neutralization

## Executive Summary

This framework provides a comprehensive prompting strategy for the BiasNeutralize AI system, optimizing the four identified prompt types for consistent, unbiased, and culturally-aware LLM outputs. The framework addresses prompt architecture, safety protocols, cultural sensitivity, and implementation strategies.

## 1. Prompt Architecture Analysis

### 1.1 Current Prompt Structure Assessment

**Identified Prompt Types:**
- **debiaser_system**: System-level role definition (stable, session-scoped)
- **debias_span**: Single-span bias neutralization
- **debias_batch**: Multi-span batch processing
- **marker_generator**: New bias-free marker generation

**Architecture Strengths:**
- Clear separation of concerns
- Modular design with reusable components
- Strict JSON output formatting
- Cultural context integration
- Intersectional bias taxonomy alignment

**Architecture Improvements Needed:**
- Enhanced error handling mechanisms
- Prompt validation and safety checks
- Dynamic prompt adaptation based on bias severity
- Improved cultural context weighting

### 1.2 Enhanced Prompt Architecture

```yaml
# Enhanced prompt architecture with safety layers
prompt_architecture:
  layers:
    - safety_layer: "Input validation and content filtering"
    - context_layer: "Cultural and linguistic context processing"
    - analysis_layer: "Bias detection and severity assessment"
    - generation_layer: "Neutralization and rewriting"
    - validation_layer: "Output quality and safety verification"
    - self_check_layer: "Epistemic classification and confidence adjustment"
```

## 2. Prompt Engineering Optimization Strategy

### 2.1 Variable and Placeholder Optimization

**Critical Variables Identified:**
```yaml
core_variables:
  # Language and Cultural Context
  input_language: "{{input_language}}"
  output_language: "{{output_language}}"
  sender_culture: "{{sender_culture}}"
  receiver_culture: "{{receiver_culture}}"

  # Content and Context
  bias_span: "{{bias_span}}"
  full_sentence_or_paragraph: "{{full_sentence_or_paragraph}}"
  context_topic: "{{context_topic}}"
  audience: "{{audience}}"
  formality_level: "{{formality_level}}"

  # Bias Metadata
  bias_family: "{{bias_family}}"
  bias_subtype: "{{bias_subtype}}"
  severity_raw: "{{severity_raw}}"
  severity_sender: "{{severity_sender}}"
  severity_receiver: "{{severity_receiver}}"
  cultural_explanation: "{{cultural_explanation}}"
```

**Enhanced Variable Validation:**
```yaml
variable_validation:
  input_language:
    type: "string"
    regex: "^[a-z]{2}(-[A-Z]{2})?$"
    required: true
    fallback: "en"

  severity_raw:
    type: "float"
    range: [0.0, 10.0]
    required: true
    validation: "Must be numeric between 0-10"

  bias_family:
    type: "enum"
    values: ["Racism", "Sexism", "Classism", "Ableism", "Ageism", "Queerfeindlichkeit", "Xenophobie", "Religious", "Other"]
    required: true
```

### 2.2 JSON Output Structure Requirements

**Enhanced JSON Schema:**
```json
{
  "output_schema": {
    "type": "object",
    "required": ["span_id", "language", "bias_analysis", "neutralization", "quality_metrics"],
    "properties": {
      "span_id": {"type": "string"},
      "language": {"type": "string"},
      "bias_analysis": {
        "type": "object",
        "properties": {
          "bias_family": {"type": "string"},
          "bias_subtype": {"type": "string"},
          "severity_assessment": {
            "type": "object",
            "properties": {
              "raw_severity": {"type": "number", "minimum": 0, "maximum": 10},
              "cultural_adjusted": {
                "sender": {"type": "number"},
                "receiver": {"type": "number"}
              },
              "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            }
          },
          "cultural_context": {
            "type": "object",
            "properties": {
              "explanation": {"type": "string"},
              "hofstede_factors": {"type": "object"},
              "risk_assessment": {"type": "string"}
            }
          }
        }
      },
      "neutralization": {
        "type": "object",
        "properties": {
          "can_preserve_core_intent": {"type": "boolean"},
          "variant_A_neutral": {
            "type": "object",
            "properties": {
              "text": {"type": "string"},
              "rationale": {"type": "string"},
              "epistemic_markers": {"type": "array"}
            }
          },
          "variant_B_emotional": {
            "type": "object",
            "properties": {
              "text": {"type": "string"},
              "rationale": {"type": "string"},
              "epistemic_markers": {"type": "array"}
            }
          }
        }
      },
      "quality_metrics": {
        "type": "object",
        "properties": {
          "self_bias_check": {
            "fact_claims": {"type": "number"},
            "logical_claims": {"type": "number"},
            "subjective_claims": {"type": "number"},
            "overconfidence_flags": {"type": "array"}
          },
          "safety_flags": {"type": "array"},
          "cultural_appropriateness": {"type": "number"}
        }
      }
    }
  }
}
```

### 2.3 Cultural Context Integration

**Enhanced Cultural Framework:**
```yaml
cultural_integration:
  hofstede_dimensions:
    power_distance: "Authority and hierarchy sensitivity"
    individualism: "Individual vs collective focus"
    masculinity: "Competition vs cooperation orientation"
    uncertainty_avoidance: "Tolerance for ambiguity"
    long_term_orientation: "Tradition vs adaptation"
    indulgence: "Gratification vs restraint"

  cultural_risk_factors:
    high_context_cultures: ["JP", "CN", "KR", "AE"]
    direct_communication_cultures: ["DE", "NL", "DK"]
    hierarchical_cultures: ["MY", "IN", "PH"]

  severity_adjustments:
    cross_cultural_multipliers:
      "DE->JP":
        directness_penalty: 1.3
        hierarchy_sensitivity: 1.5
      "US->CN":
        individualism_clash: 1.2
        face_saving: 1.4
```

## 3. Safety and Bias Mitigation Strategy

### 3.1 Self-Bias Check Integration Requirements

**Enhanced Self-Bias Check Protocol:**
```yaml
self_bias_check:
  epistemic_classification:
    factual_markers:
      prefixes: ["Faktisch korrekt", "Nachweislich", "Dokumentiert"]
      confidence_threshold: 0.85
      verification_required: true

    logical_markers:
      prefixes: ["Logisch scheint", "Daraus folgt", "Plausibel ist"]
      confidence_threshold: 0.70
      reasoning_chain_required: true

    subjective_markers:
      prefixes: ["Rein subjektiv", "Aus meiner Sicht", "Möglicherweise"]
      confidence_threshold: 0.50
      perspective_acknowledgment: true

  overconfidence_reduction:
    triggers:
      - confidence_score: "> 0.95"
      - absolute_language: ["immer", "nie", "alle", "niemand"]
      - certainty_claims: ["zweifellos", "eindeutig", "sicher"]

    mitigations:
      - add_hedging: ["oft", "in der Regel", "tendenziell"]
      - add_uncertainty: ["scheint", "könnte", "möglicherweise"]
      - add_context: ["im gegebenen Kontext", "unter den Umständen"]
```

### 3.2 Cultural Sensitivity Considerations

**Cultural Safety Protocols:**
```yaml
cultural_sensitivity:
  high_risk_scenarios:
    - cross_cultural_directness: "DE->JP, NL->CN"
    - hierarchy_violations: "egalitarian->hierarchical"
    - religious_references: "secular->religious contexts"
    - gender_role_assumptions: "progressive->conservative"

  mitigation_strategies:
    hedging_language:
      - "in some cultural contexts"
      - "from a [sender_culture] perspective"
      - "considering cultural differences"

    cultural_bridging:
      - explain_cultural_rationale: true
      - provide_alternative_framings: true
      - acknowledge_perspective_differences: true

  forbidden_operations:
    - cultural_stereotyping: "Never reinforce cultural stereotypes"
    - cultural_ranking: "Never rank cultures as superior/inferior"
    - cultural_erasure: "Never ignore legitimate cultural differences"
```

### 3.3 Fact vs. Opinion Classification

**Enhanced Classification Framework:**
```yaml
classification_framework:
  fact_identification:
    verifiable_claims:
      - statistical_data: "Numbers, percentages, measurements"
      - documented_events: "Historical facts, recorded incidents"
      - scientific_findings: "Peer-reviewed research, studies"

    verification_requirements:
      - source_attribution: "Must cite or reference sources"
      - temporal_context: "Must specify time period"
      - scope_limitation: "Must define applicable context"

  opinion_identification:
    subjective_indicators:
      - value_judgments: "good/bad, right/wrong"
      - preference_statements: "should/shouldn't, prefer"
      - interpretive_claims: "means, suggests, implies"

    explicit_marking:
      - opinion_prefixes: ["In my view", "I believe", "It appears"]
      - uncertainty_markers: ["possibly", "likely", "seems"]
      - perspective_qualifiers: ["from this perspective", "in this context"]
```

## 4. Implementation Recommendations

### 4.1 Optimal Prompt Chaining Strategies

**Sequential Processing Chain:**
```yaml
prompt_chain:
  stage_1_validation:
    purpose: "Input safety and format validation"
    prompts: ["input_validator", "content_safety_check"]
    fallback: "sanitize_and_retry"

  stage_2_analysis:
    purpose: "Bias detection and cultural assessment"
    prompts: ["bias_analyzer", "cultural_assessor"]
    parallel: true

  stage_3_generation:
    purpose: "Neutralization and rewriting"
    prompts: ["neutralizer_A", "neutralizer_B"]
    context_injection: "bias_analysis + cultural_context"

  stage_4_validation:
    purpose: "Quality and safety verification"
    prompts: ["self_bias_checker", "output_validator"]
    quality_gates: ["safety_score > 0.8", "coherence_score > 0.7"]
```

**Parallel Processing Optimization:**
```yaml
parallel_optimization:
  independent_processes:
    - cultural_analysis: "Can run parallel to bias detection"
    - variant_generation: "A and B variants can be generated simultaneously"
    - quality_metrics: "Can be computed during generation"

  resource_allocation:
    high_priority: ["safety_checks", "bias_detection"]
    medium_priority: ["cultural_analysis", "variant_generation"]
    low_priority: ["formatting", "metadata_enrichment"]
```

### 4.2 Error Handling and Fallback Mechanisms

**Comprehensive Error Handling:**
```yaml
error_handling:
  input_errors:
    invalid_language:
      detection: "Language code validation"
      fallback: "Default to English with warning"

    malformed_content:
      detection: "Content structure validation"
      fallback: "Request content reformat"

    missing_cultural_context:
      detection: "Cultural profile availability"
      fallback: "Use neutral cultural baseline"

  processing_errors:
    bias_detection_failure:
      detection: "No bias categories detected"
      fallback: "Manual review flag + basic safety check"

    cultural_analysis_failure:
      detection: "Cultural profile lookup failure"
      fallback: "Skip cultural adjustment with notification"

    llm_generation_failure:
      detection: "Empty or malformed LLM response"
      fallback: "Retry with simplified prompt"

  output_errors:
    json_validation_failure:
      detection: "JSON schema validation"
      fallback: "Extract and restructure valid components"

    safety_check_failure:
      detection: "Safety score below threshold"
      fallback: "Flag for human review + safe default response"
```

### 4.3 Performance Optimization Approaches

**Latency Optimization:**
```yaml
performance_optimization:
  caching_strategy:
    cultural_profiles: "Cache frequently used profiles (24h TTL)"
    bias_patterns: "Cache pattern matching results (1h TTL)"
    llm_responses: "Cache similar content hashes (30min TTL)"

  batch_processing:
    optimal_batch_size: 5-10  # spans per batch
    batching_criteria:
      - same_document: "Group spans from same document"
      - similar_severity: "Group spans with similar severity"
      - same_bias_family: "Group spans with same bias type"

  model_optimization:
    lightweight_models:
      bias_detection: "Use DistilBERT for initial screening"
      language_detection: "Use fastText for language identification"

    heavy_models:
      neutralization: "Use GPT-4/Claude for complex rewriting"
      cultural_analysis: "Use full models for cultural nuances"
```

### 4.4 Quality Assurance Checkpoints

**Multi-Level Quality Gates:**
```yaml
quality_assurance:
  checkpoint_1_input:
    validations:
      - content_safety: "Screen for obviously harmful content"
      - format_compliance: "Validate input structure"
      - language_detection: "Confirm language identification"

    failure_handling: "Reject with specific error message"

  checkpoint_2_analysis:
    validations:
      - bias_detection_confidence: "Minimum confidence threshold"
      - cultural_profile_availability: "Verify cultural context"
      - severity_assessment_consistency: "Cross-validate severity scores"

    failure_handling: "Flag for manual review or simplified processing"

  checkpoint_3_generation:
    validations:
      - variant_quality: "Assess neutralization effectiveness"
      - cultural_appropriateness: "Verify cultural sensitivity"
      - intent_preservation: "Confirm core intent maintained"

    failure_handling: "Regenerate with adjusted parameters"

  checkpoint_4_output:
    validations:
      - self_bias_compliance: "Verify epistemic markers present"
      - json_schema_compliance: "Validate output structure"
      - safety_score: "Final safety assessment"

    failure_handling: "Apply safe defaults and flag for review"
```

## 5. Advanced Prompt Engineering Techniques

### 5.1 Dynamic Prompt Adaptation

**Context-Aware Prompt Selection:**
```yaml
adaptive_prompting:
  severity_based_adaptation:
    low_severity: "0-3"
      prompt_style: "gentle_neutralization"
      safety_level: "standard"

    medium_severity: "4-7"
      prompt_style: "balanced_rewriting"
      safety_level: "enhanced"

    high_severity: "8-10"
      prompt_style: "strong_intervention"
      safety_level: "maximum"

  cultural_distance_adaptation:
    low_distance: "Same cultural cluster"
      cultural_explanation: "minimal"
      adjustment_factor: 1.0

    medium_distance: "Different but familiar cultures"
      cultural_explanation: "moderate"
      adjustment_factor: 1.2

    high_distance: "Very different cultural contexts"
      cultural_explanation: "detailed"
      adjustment_factor: 1.5
```

### 5.2 Prompt Validation and Testing

**Validation Framework:**
```yaml
prompt_validation:
  consistency_testing:
    same_input_multiple_runs:
      runs: 5
      variance_threshold: 0.1
      metrics: ["severity_score", "variant_quality"]

    similar_inputs:
      test_set_size: 100
      similarity_threshold: 0.8
      expected_consistency: 0.85

  bias_testing:
    prompt_bias_detection:
      test_categories: ["gender", "race", "culture", "religion"]
      bias_threshold: 0.05
      mitigation_required: true

    cultural_bias_testing:
      test_cultures: ["DE", "JP", "US", "CN", "BR", "IN"]
      fairness_metrics: ["equal_treatment", "appropriate_sensitivity"]

  safety_testing:
    adversarial_inputs:
      harmful_content: "Test resistance to toxic inputs"
      manipulation_attempts: "Test resistance to prompt injection"
      edge_cases: "Test handling of unusual inputs"
```

## 6. Implementation Roadmap

### 6.1 Phase 1: Core Framework (Days 1-3)
- Implement enhanced prompt templates
- Set up validation and safety layers
- Create basic error handling

### 6.2 Phase 2: Cultural Integration (Days 4-5)
- Integrate Hofstede cultural profiles
- Implement cultural sensitivity protocols
- Add cultural explanation generation

### 6.3 Phase 3: Quality Assurance (Days 6-7)
- Implement self-bias check system
- Add comprehensive quality gates
- Create monitoring and metrics

### 6.4 Phase 4: Optimization (Ongoing)
- Performance tuning and caching
- Advanced prompt adaptation
- Continuous quality improvement

## 7. Success Metrics and KPIs

```yaml
success_metrics:
  quality_metrics:
    bias_detection_f1: ">= 0.85"
    cultural_appropriateness: ">= 0.80"
    intent_preservation: ">= 0.85"
    safety_score: ">= 0.95"

  performance_metrics:
    response_latency: "< 5 seconds (median)"
    error_rate: "< 5%"
    cache_hit_ratio: "> 60%"

  user_satisfaction:
    expert_approval_rate: ">= 80%"
    cultural_sensitivity_score: ">= 85%"
    usability_rating: ">= 4.0/5.0"
```

## Conclusion

This optimal prompting framework provides a comprehensive approach to bias detection and neutralization that prioritizes safety, cultural sensitivity, and quality while maintaining high performance. The modular design allows for iterative improvement and adaptation to new requirements while ensuring consistent, reliable outputs across diverse cultural and linguistic contexts.