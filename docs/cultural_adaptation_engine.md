# Cultural Adaptation Engine

## Overview

The Cultural Adaptation Engine is a sophisticated system that enhances bias detection with cross-cultural intelligence based on Hofstede's 6-Dimensional cultural model. It analyzes cultural differences between communication participants and adjusts bias severity scores accordingly, providing contextual explanations and mitigation strategies.

## Architecture

### Core Components

```
cultural/
├── models/
│   └── hofstede_model.py      # Hofstede 6D cultural dimensions model
├── data/
│   ├── cultural_profiles.json  # Cultural profiles database
│   └── profile_manager.py      # Profile loading and caching
├── adapters/
│   └── cultural_adapter.py     # Bias severity adjustment engine
├── analyzers/
│   └── cultural_analyzer.py    # Cross-cultural context analysis
├── intelligence/
│   └── cultural_intelligence.py # Cultural intelligence features
└── integration.py              # Integration hooks and interfaces
```

## Features

### 1. Hofstede Cultural Model Implementation

The system implements all 6 dimensions of Hofstede's cultural model:

- **PDI (Power Distance Index)**: Acceptance of power inequality
- **IDV (Individualism vs Collectivism)**: Individual vs group orientation
- **MAS (Masculinity vs Femininity)**: Achievement vs relationship focus
- **UAI (Uncertainty Avoidance Index)**: Tolerance for ambiguity
- **LTO (Long Term Orientation)**: Time perspective and tradition
- **IVR (Indulgence vs Restraint)**: Expression and gratification control

### 2. Cultural Distance Calculation

```python
from bias_engine.cultural import HofstedeModel

model = HofstedeModel()
distance = model.calculate_cultural_distance(
    culture1_dimensions,
    culture2_dimensions,
    use_weights=True
)
```

### 3. Bias Severity Adjustment

The system adjusts bias severity scores based on cultural context:

```python
from bias_engine.cultural import CulturalAdapter

adapter = CulturalAdapter()
enhanced_results = adapter.apply_cultural_modifiers(
    bias_results,
    sender_culture="DE",
    receiver_culture="US"
)
```

### 4. Cultural Intelligence Features

- **Radar Chart Generation**: Visual representation of cultural dimensions
- **Communication Style Analysis**: Direct/indirect, formal/informal patterns
- **Sensitivity Warnings**: Alerts for high-risk cultural differences
- **Bridge Building Suggestions**: Strategies to overcome cultural gaps

## Supported Cultures

The engine currently supports:

| Code | Country | Cultural Cluster |
|------|---------|------------------|
| DE   | Germany | Germanic |
| US   | United States | Anglo |
| JP   | Japan | East Asian |
| CN   | China | East Asian |
| FR   | France | Latin |
| IT   | Italy | Latin |
| ES   | Spain | Latin |

### Adding New Cultures

Add new cultural profiles to `cultural_profiles.json`:

```json
{
  "NEW": {
    "country": "New Country",
    "code": "NEW",
    "dimensions": {
      "PDI": 50, "IDV": 50, "MAS": 50,
      "UAI": 50, "LTO": 50, "IVR": 50
    },
    "characteristics": {
      "communication_style": "moderate",
      "hierarchy_acceptance": "medium"
    }
  }
}
```

## Integration with Bias Detection

### Simple Enhancement

```python
from bias_engine.cultural import enhance_bias_with_culture

# Original bias detection results
bias_results = {
    "overall_bias_score": 0.7,
    "biases_detected": {
        "gender": {"severity": 0.8, "confidence": 0.9}
    }
}

# Enhance with cultural context
enhanced_results = enhance_bias_with_culture(
    bias_results,
    sender_culture="DE",
    receiver_culture="JP",
    context={"type": "business"}
)
```

### Decorator-based Integration

```python
from bias_engine.cultural import cultural_bias_enhancer

@cultural_bias_enhancer
def detect_bias(text, sender_culture=None, receiver_culture=None, context=None):
    # Your bias detection logic here
    return bias_results
```

### Full Integration

```python
from bias_engine.cultural import CulturalIntegration

integration = CulturalIntegration()

# Register hooks
def pre_analysis_hook(bias_results, sender, receiver, context):
    print(f"Analyzing {sender} -> {receiver}")

integration.register_pre_bias_hook(pre_analysis_hook)

# Enhanced analysis
enhanced_results = integration.enhance_bias_detection(
    bias_results, "DE", "US", {"type": "educational"}
)
```

## Cultural Context Analysis

### Risk Assessment

The system provides comprehensive risk assessment:

```python
from bias_engine.cultural import get_cultural_context

context = get_cultural_context("DE", "JP", {"type": "business"})

print(f"Cultural distance: {context['cultural_distance']}")
print(f"Risk level: {context['overall_risk']}")
print(f"Recommendations: {context['recommendations']}")
```

### Communication Strategies

```python
from bias_engine.cultural import CulturalIntelligence

ci = CulturalIntelligence()
strategies = ci.recommend_communication_strategies("DE", "JP", "business")

print("Sender adaptations:", strategies["sender_adaptations"])
print("General strategies:", strategies["general_strategies"])
```

## Dashboard Integration

Generate dashboard-ready data:

```python
dashboard_data = integration.get_cultural_dashboard_data(["DE", "US", "JP", "CN"])

# Includes:
# - Radar chart data
# - Cultural distances matrix
# - Communication styles
# - Sensitivity warnings
# - Recommendations
```

## Performance and Caching

### Caching Strategy

The system uses intelligent caching:

- **Profile Caching**: Cultural profiles cached with LRU eviction
- **Result Caching**: Analysis results cached by parameters
- **Configurable**: Caching can be disabled for real-time scenarios

### Performance Optimization

```python
# Enable caching for better performance
integration = CulturalIntegration(enable_caching=True)

# Batch processing for multiple culture pairs
cultures = ["DE", "US", "JP", "CN"]
dashboard_data = integration.get_cultural_dashboard_data(cultures)
```

## Error Handling

The system provides robust error handling:

```python
try:
    enhanced_results = enhance_bias_with_culture(bias_results, "INVALID", "US")
except CulturalIntegrationError as e:
    print(f"Cultural enhancement failed: {e}")
    # Fall back to original results
    enhanced_results = bias_results
```

## Example Use Cases

### 1. International Business Communication

```python
# Email bias analysis between German and Japanese colleagues
bias_results = analyze_email_bias(email_content)
enhanced_results = enhance_bias_with_culture(
    bias_results,
    sender_culture="DE",
    receiver_culture="JP",
    context={"type": "business", "urgency": "high"}
)
```

### 2. Educational Content Review

```python
# Course material bias analysis for diverse student body
bias_results = analyze_content_bias(course_material)
enhanced_results = enhance_bias_with_culture(
    bias_results,
    sender_culture="US",
    receiver_culture="CN",
    context={"type": "educational", "level": "university"}
)
```

### 3. Healthcare Communication

```python
# Patient-doctor communication bias analysis
bias_results = analyze_consultation_bias(conversation)
enhanced_results = enhance_bias_with_culture(
    bias_results,
    sender_culture="DE",
    receiver_culture="ES",
    context={"type": "healthcare", "sensitive": True}
)
```

## Configuration

### Environment Variables

```bash
# Optional: Custom cultural profiles path
CULTURAL_PROFILES_PATH=/path/to/custom/profiles.json

# Optional: Enable debug logging
CULTURAL_DEBUG=true

# Optional: Cache settings
CULTURAL_CACHE_SIZE=1000
CULTURAL_CACHE_TTL=3600
```

### Custom Configuration

```python
from bias_engine.cultural import CulturalProfileManager, CulturalIntegration

# Custom profile manager
profile_manager = CulturalProfileManager("/path/to/profiles.json")

# Custom integration
integration = CulturalIntegration(
    profile_manager=profile_manager,
    enable_caching=True
)
```

## API Reference

### Core Classes

- **`HofstedeModel`**: Cultural dimensions analysis and distance calculation
- **`CulturalAdapter`**: Bias severity adjustment with cultural context
- **`CulturalAnalyzer`**: Cross-cultural communication risk analysis
- **`CulturalIntelligence`**: Advanced cultural intelligence features
- **`CulturalIntegration`**: Main integration interface

### Key Methods

- **`calculate_cultural_distance()`**: Compute distance between cultures
- **`apply_cultural_modifiers()`**: Adjust bias scores with cultural context
- **`analyze_cross_cultural_context()`**: Comprehensive cultural analysis
- **`recommend_communication_strategies()`**: Generate communication strategies
- **`enhance_bias_detection()`**: Main enhancement entry point

## Best Practices

### 1. Culture Code Validation

Always validate culture codes before analysis:

```python
if integration.validate_culture_codes("DE", "US"):
    enhanced_results = integration.enhance_bias_detection(...)
else:
    # Handle invalid codes
```

### 2. Context Enrichment

Provide context information for better analysis:

```python
context = {
    "type": "business",  # business, educational, healthcare
    "urgency": "high",   # low, medium, high
    "formality": "formal" # formal, informal, mixed
}
```

### 3. Error Handling

Implement proper error handling:

```python
try:
    enhanced_results = enhance_bias_with_culture(...)
except CulturalIntegrationError:
    # Fall back to original results
    enhanced_results = original_bias_results
```

### 4. Performance Monitoring

Monitor performance with statistics:

```python
stats = integration.get_cultural_statistics()
print(f"Cache hit rate: {stats['cache_size']}")
print(f"Supported cultures: {stats['supported_cultures']}")
```

## Troubleshooting

### Common Issues

1. **Missing Culture Code**: Ensure culture codes are in supported list
2. **Cache Issues**: Clear cache if stale results: `integration.clear_cache()`
3. **Profile Loading**: Check profile file path and permissions
4. **Memory Usage**: Disable caching for memory-constrained environments

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger('bias_engine.cultural').setLevel(logging.DEBUG)
```

## Future Enhancements

- **Dynamic Profile Learning**: Learn cultural patterns from user feedback
- **Regional Variations**: Support for regional cultural variations
- **Industry-Specific Profiles**: Specialized profiles for different industries
- **Real-time Adaptation**: Dynamic adjustment based on interaction patterns
- **Multi-cultural Teams**: Analysis for teams with diverse cultural backgrounds