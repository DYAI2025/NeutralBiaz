# LLM Prompting Framework Documentation

This directory contains comprehensive documentation for the optimal LLM prompting strategy designed for the BiasNeutralize AI system.

## 📁 Documentation Structure

### Core Framework Documents

- **[`optimal_prompting_framework.md`](./optimal_prompting_framework.md)** - Main framework specification with analysis, strategies, and recommendations
- **[`enhanced_prompt_templates.yaml`](./enhanced_prompt_templates.yaml)** - Production-ready prompt templates with safety enhancements
- **[`implementation_guide.md`](./implementation_guide.md)** - Concrete implementation patterns and code examples

## 🎯 Key Features

### Enhanced Safety Protocols
- Multi-layer validation system
- Comprehensive self-bias checking
- Cultural sensitivity safeguards
- Epistemic classification requirements

### Cultural Intelligence
- Hofstede-based cultural adaptation
- Cross-cultural risk assessment
- Cultural bridge explanations
- Dynamic severity adjustments

### Quality Assurance
- Automated testing frameworks
- Performance monitoring systems
- Error handling and recovery
- Continuous validation pipelines

## 🚀 Quick Start

### 1. Framework Overview
The optimal prompting framework consists of four enhanced prompt types:

```yaml
Prompt Types:
  - debiaser_system_enhanced: System-level role with safety protocols
  - debias_span_enhanced: Single-span analysis with cultural awareness
  - debias_batch_enhanced: Multi-span processing with consistency
  - marker_generator_enhanced: Safe marker generation with validation
  - self_bias_check_enhanced: Comprehensive output validation
```

### 2. Implementation Pattern

```python
# Basic usage pattern
from bias_neutralizer import BiasNeutralizationPipeline

pipeline = BiasNeutralizationPipeline(
    llm_client=your_llm_client,
    prompt_manager=PromptManager("enhanced_prompt_templates.yaml")
)

result = await pipeline.process_span({
    "input_language": "de",
    "output_language": "en",
    "bias_span": "problematic text",
    "sender_culture": "DE",
    "receiver_culture": "US",
    # ... other context
})
```

### 3. Key Configuration

```yaml
# Essential configuration parameters
safety_thresholds:
  minimum_safety_score: 0.80
  maximum_bias_risk: 0.20
  cultural_appropriateness_threshold: 0.70

validation_requirements:
  epistemic_marker_compliance: 90%
  self_bias_check: mandatory
  cultural_risk_assessment: required
```

## 📊 Performance Metrics

### Target KPIs
- **Bias Detection F1**: ≥ 0.85
- **Cultural Appropriateness**: ≥ 0.80
- **Safety Score**: ≥ 0.95
- **Response Latency**: < 5 seconds (median)
- **Expert Approval Rate**: ≥ 80%

### Quality Gates
- Input validation with sanitization
- Cultural risk assessment
- Multi-layer output validation
- Self-bias compliance checking
- Safety score verification

## 🛡️ Safety Features

### Input Safety
- Content validation and sanitization
- Prompt injection protection
- Variable substitution security
- Cultural context verification

### Processing Safety
- Bias amplification prevention
- Cultural sensitivity enforcement
- Intent preservation validation
- Harm reduction protocols

### Output Safety
- Comprehensive self-bias checking
- Epistemic classification verification
- Cultural appropriateness validation
- Final safety score assessment

## 🌍 Cultural Intelligence

### Supported Cultural Frameworks
- **Hofstede 6D**: Power Distance, Individualism, Masculinity, Uncertainty Avoidance, Long-term Orientation, Indulgence
- **Cultural Risk Patterns**: Directness, Hierarchy, Collectivism considerations
- **Cross-Cultural Bridging**: Explanatory mechanisms for cultural differences

### Cultural Safety Protocols
```yaml
High-Risk Scenarios:
  - Direct → Indirect cultures (DE→JP)
  - Egalitarian → Hierarchical contexts
  - Individual → Collective orientations
  - Secular → Religious contexts

Mitigation Strategies:
  - Cultural bridge explanations
  - Hedging language adaptation
  - Context-appropriate reformulation
  - Risk factor identification
```

## 🔧 Implementation Components

### Core Classes
- `PromptManager`: Template management and variable substitution
- `BiasNeutralizationPipeline`: End-to-end processing workflow
- `CulturalContextManager`: Cultural profile and risk assessment
- `ComprehensiveValidator`: Multi-layer output validation

### Integration Points
- LLM client abstraction layer
- Cultural profile database
- Validation rule engine
- Monitoring and metrics system

## 📈 Testing & Evaluation

### Automated Testing Suite
- Unit tests for individual prompt types
- Integration tests for full workflows
- Cultural bias detection tests
- Performance regression tests

### Evaluation Methodologies
- Expert human evaluation protocols
- Cross-cultural fairness assessment
- Systematic bias pattern detection
- Quality consistency validation

## 🚨 Error Handling

### Graceful Degradation
- Input validation failure recovery
- Cultural profile fallbacks
- Generation failure alternatives
- Validation error mitigation

### Monitoring & Alerting
- Real-time performance tracking
- Quality score monitoring
- Error pattern detection
- Automated alert systems

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] Prompt templates validated
- [ ] Cultural profiles loaded
- [ ] Safety thresholds configured
- [ ] Test suite passing
- [ ] Monitoring systems ready

### Production Requirements
- [ ] LLM provider configured
- [ ] Caching system enabled
- [ ] Error handling active
- [ ] Metrics collection running
- [ ] Human review workflows established

## 🔄 Maintenance & Updates

### Regular Tasks
- Prompt performance analysis
- Cultural profile updates
- Safety threshold adjustments
- Test case expansion
- Quality metric review

### Improvement Cycles
- A/B testing of prompt variations
- Cultural bias pattern analysis
- Performance optimization
- Safety protocol enhancement
- User feedback integration

## 📞 Support & Resources

### Documentation References
- [Bias Taxonomy v5](../bias-taxonomy-v5-intersectional.md)
- [Cultural Models Integration](../cultural-models/)
- [Safety Protocols](../safety-protocols.md)
- [Performance Guidelines](../performance-optimization.md)

### Key Contacts
- Technical Lead: Bias Detection Team
- Cultural Expert: Cross-Cultural AI Safety
- Safety Officer: AI Ethics & Safety
- Product Owner: Trust & Safety Product

---

## 🎯 Next Steps

1. **Review Framework**: Start with `optimal_prompting_framework.md`
2. **Examine Templates**: Study `enhanced_prompt_templates.yaml`
3. **Implementation**: Follow `implementation_guide.md`
4. **Testing**: Set up evaluation frameworks
5. **Deployment**: Configure production systems

For questions or support, consult the implementation guide or contact the development team.