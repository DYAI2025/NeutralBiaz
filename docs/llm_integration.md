# LLM Integration for Bias Detection and Neutralization

## Overview

The LLM integration provides advanced bias detection and neutralization capabilities using Large Language Models. It implements the German debiasing specification with epistemic classification and cultural context awareness.

## Key Features

### 1. **Multi-Provider LLM Support**
- **OpenAI GPT-4**: High-quality responses with good multilingual support
- **Anthropic Claude**: Excellent safety features and reasoning capabilities
- **Azure OpenAI**: Enterprise-grade deployment with security features
- **Google Gemini**: Advanced multimodal capabilities (future support)

### 2. **Prompt Management System**
- YAML-based template configuration
- Variable substitution with validation
- Built-in German specification prompts:
  - `debiaser_system`: System role definition
  - `debias_span`: Single span neutralization
  - `debias_batch`: Batch processing
  - `marker_generator`: Bias marker creation
  - `self_bias_check`: Epistemic classification

### 3. **Debiasing Pipeline**
- **A/B Variant Generation**:
  - Variant A: Neutral, factual alternatives
  - Variant B: Emotionally similar but bias-reduced
- **Cultural Context Integration**: Adjusts severity based on cultural backgrounds
- **Quality Validation**: Ensures response quality and safety

### 4. **Self-Bias Check System**
- **Epistemic Classification**:
  - `faktisch`: Objective, verifiable statements
  - `logisch`: Rational conclusions and arguments
  - `subjektiv`: Opinions and personal assessments
- **Overconfidence Detection**: Identifies and reduces overconfident language
- **Prefix Enforcement**: Adds German epistemic prefixes to outputs

### 5. **Cultural Context System**
- Cross-cultural communication analysis
- Severity adjustments based on sender/receiver cultures
- Cultural communication recommendations
- Support for: DE, US, JP, GB, FR, ES, and neutral contexts

## API Endpoints

### Single Span Debiasing
```http
POST /api/v1/llm/debias/single
Content-Type: application/json

{
  "bias_detection": {
    "type": "racial",
    "level": "high",
    "confidence": 0.85,
    "description": "Racial stereotype detected",
    "affected_text": "problematic phrase",
    "start_position": 20,
    "end_position": 39,
    "suggestions": []
  },
  "full_text": "This is a test text with problematic phrase that needs fixing.",
  "input_language": "en",
  "output_language": "de",
  "sender_culture": "de",
  "receiver_culture": "us",
  "context_topic": "workplace communication",
  "audience": "professional",
  "formality_level": "formal"
}
```

### Batch Span Debiasing
```http
POST /api/v1/llm/debias/batch
Content-Type: application/json

{
  "bias_detections": [...],
  "full_text": "Complete document text...",
  "input_language": "de",
  "output_language": "de",
  "sender_culture": "de",
  "receiver_culture": "neutral",
  "context_topic": "public communication",
  "audience": "general",
  "formality_level": "neutral"
}
```

### Marker Generation
```http
POST /api/v1/llm/markers/generate
Content-Type: application/json

{
  "bias_family": "racism",
  "bias_subtype": "stereotyping",
  "bias_description": "Stereotypical assumptions about racial groups",
  "output_language": "de",
  "domain": "social_media",
  "primary_cultures": ["de", "us"]
}
```

### Self-Bias Checking
```http
POST /api/v1/llm/self-bias/check
Content-Type: application/json

{
  "texts": [
    "Das ist definitiv die beste Lösung.",
    "Nach der Studie von 2023 sind 70% betroffen.",
    "Ich finde, dass dies eine interessante Perspektive ist."
  ],
  "context": "Technical discussion",
  "llm_provider": "anthropic"
}
```

## Configuration

### Environment Variables

```bash
# Default LLM provider
LLM_DEFAULT_PROVIDER=anthropic

# Provider API keys
ANTHROPIC_API_KEY=your-api-key
OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Rate limiting
LLM_RATE_LIMIT_ENABLED=true
LLM_RATE_LIMIT_RPM=60
LLM_RATE_LIMIT_TPM=100000

# Quality settings
LLM_RESPONSE_VALIDATION=true
LLM_QUALITY_THRESHOLD=0.7
```

### Custom Prompt Templates

Create `config/prompts.yaml`:
```yaml
custom_template:
  role: user
  description: Custom template description
  content: |
    Your custom prompt with {{variables}}.

    Context: {{context}}
    Task: {{task}}

    Respond in {{language}}.
```

## Usage Examples

### Python Integration

```python
from bias_engine.llm import DebiasingPipeline, LLMConfig, LLMProvider

# Configure LLM
config = LLMConfig(
    provider=LLMProvider.ANTHROPIC,
    api_key="your-api-key",
    model="claude-3-sonnet-20240229"
)

# Create pipeline
async with DebiasingPipeline(config) as pipeline:
    # Process bias detection
    result = await pipeline.debias_span(request)

    print(f"Variant A: {result.variant_a_rewrite}")
    print(f"Variant B: {result.variant_b_rewrite}")
```

### Cultural Context Integration

```python
from bias_engine.llm.cultural_integration import CulturalContextIntegrator

integrator = CulturalContextIntegrator()

# Get cultural recommendations
recommendations = integrator.get_cultural_recommendations("de", "jp")
print(recommendations["recommendations"])

# Calculate cultural severity
severity = integrator.calculate_cultural_severity(
    bias_type="racism",
    raw_severity=8.0,
    sender_culture="de",
    receiver_culture="us"
)
```

### Self-Bias Checking

```python
from bias_engine.llm import SelfBiasChecker

checker = SelfBiasChecker(llm_client)

# Check single text
request = SelfBiasCheckRequest(
    text="Das ist definitiv richtig.",
    context="Technical discussion"
)

result = await checker.check_bias(request)
print(f"Classification: {result.epistemic_classification}")
print(f"Corrected: {result.corrected_text}")
```

## Response Examples

### Debiasing Response
```json
{
  "success": true,
  "result": {
    "span_id": "span_20_39",
    "language": "de",
    "bias_family": "racism",
    "bias_subtype": "stereotyping",
    "analysis_explanation": "Der Begriff verstärkt negative Stereotype über eine Bevölkerungsgruppe...",
    "can_preserve_core_intent": true,
    "variant_A_rewrite": "Menschen mit unterschiedlichen kulturellen Hintergründen",
    "variant_B_rewrite": "Diese vielfältige Gruppe von Menschen",
    "safety_notes": "Ursprüngliche Kritik wurde beibehalten, aber respektvoller formuliert",
    "confidence_score": 0.85
  },
  "quality_issues": []
}
```

### Self-Bias Check Response
```json
{
  "success": true,
  "results": [
    {
      "original_text": "Das ist definitiv die beste Lösung.",
      "epistemic_classification": "subjektiv",
      "overconfidence_detected": true,
      "bias_indicators": ["definitiv"],
      "corrected_text": "Rein subjektiv, aus meinem Denken ergibt sich, dass das eine gute Lösung sein könnte.",
      "confidence_score": 0.8,
      "explanation": "Überzeugungssprache ohne Belege deutet auf subjektive Meinung hin."
    }
  ],
  "statistics": {
    "total_processed": 1,
    "overconfidence_detected": 1,
    "classifications": {
      "faktisch": 0,
      "logisch": 0,
      "subjektiv": 1
    }
  }
}
```

## Quality Assurance

### Response Validation
- **JSON Format Validation**: Ensures proper response structure
- **Content Quality Checks**: Validates meaningful differences between variants
- **Safety Validation**: Checks for remaining bias or harmful content
- **Cultural Appropriateness**: Ensures cultural sensitivity

### Error Handling
- **Fallback Processing**: Rule-based alternatives when LLM fails
- **Retry Mechanisms**: Automatic retries with exponential backoff
- **Rate Limit Handling**: Intelligent request throttling
- **Provider Failover**: Automatic switching to backup providers

## Performance Considerations

### Rate Limiting
- Default: 60 requests/minute, 100K tokens/minute
- Configurable per provider
- Burst handling for traffic spikes

### Caching
- Prompt template caching
- Response caching for identical inputs
- Cultural mapping caching

### Monitoring
- Token usage tracking
- Response time monitoring
- Quality metrics collection
- Error rate monitoring

## Security

### API Key Management
- Environment variable configuration
- No hardcoded credentials
- Provider-specific security practices

### Data Privacy
- No conversation history storage
- Optional request/response logging
- GDPR compliance considerations

### Content Safety
- Built-in safety checks
- Bias amplification prevention
- Harmful content filtering

## Development and Testing

### Unit Tests
```bash
# Run LLM integration tests
pytest tests/bias_engine/llm/ -v

# Run with coverage
pytest tests/bias_engine/llm/ --cov=bias_engine.llm
```

### Integration Tests
```bash
# Test with real providers (requires API keys)
pytest tests/integration/test_llm_providers.py -v
```

### Mock Testing
All components include comprehensive mock testing for development without API costs.

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Check environment variable configuration
   - Verify API key validity and permissions
   - Ensure correct provider endpoint URLs

2. **Rate Limiting**
   - Monitor token usage in logs
   - Adjust rate limiting configuration
   - Consider provider tier upgrades

3. **Quality Issues**
   - Review prompt templates
   - Adjust quality thresholds
   - Enable detailed logging for debugging

4. **Cultural Context**
   - Verify culture codes (e.g., "de", "us", "jp")
   - Check cultural mapping configuration
   - Review severity adjustment factors

### Logging
```bash
# Enable detailed LLM logging
LLM_LOG_REQUESTS=true
LLM_LOG_RESPONSES=true  # Use carefully for privacy
LLM_LOG_TOKENS=true
LOG_LEVEL=DEBUG
```

## Future Enhancements

### Planned Features
- **Multimodal Support**: Image and video bias detection
- **Real-time Streaming**: Streaming responses for long texts
- **Custom Model Fine-tuning**: Domain-specific model adaptation
- **Advanced Metrics**: Detailed bias neutralization analytics
- **Workflow Integration**: Integration with content management systems

### Provider Roadmap
- Google Gemini integration
- Local model support (Ollama, etc.)
- Custom model endpoint support
- Federated learning capabilities

## Support

For technical issues or feature requests:
1. Check the troubleshooting section
2. Review API documentation
3. Check GitHub issues
4. Contact technical support

## License

This LLM integration follows the project's main licensing terms. See LICENSE file for details.