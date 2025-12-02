# BiazNeutralize AI - Technical Architecture Blueprint

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                           INPUT LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│  Text Input  │  Marker Files  │  Language Detection  │  Config  │
└─────────┬───────────┬─────────────────┬──────────────────┬──────┘
          │           │                 │                  │
          v           v                 v                  v
┌─────────────────────────────────────────────────────────────────┐
│                        PROCESSING ENGINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐    │
│  │ Bias Engine │  │Cultural Engine│  │   LLM Debiaser     │    │
│  │             │  │              │  │                     │    │
│  │ • spaCy     │  │ • Hofstede   │  │ • Prompt Templates  │    │
│  │ • Stanza    │  │ • GLOBE      │  │ • Variant Generator │    │
│  │ • Transform.│  │ • Hall       │  │ • Marker Creator    │    │
│  │ • Rules     │  │ • Severity   │  │ • Self-Bias Check   │    │
│  └─────────────┘  │   Adjustment │  └─────────────────────┘    │
│                   └──────────────┘                             │
└─────────┬───────────────────┬─────────────────┬──────────────────┘
          │                   │                 │
          v                   v                 v
┌─────────────────────────────────────────────────────────────────┐
│                        OUTPUT LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  JSON Export │  React Dashboard │  Cultural Viz  │  Reports     │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Component Specifications

### 1. Input Layer Components

#### 1.1 Text Processor
```python
class TextProcessor:
    def __init__(self):
        self.language_detector = fasttext.load_model('lid.176.bin')
        self.preprocessor = TextPreprocessor()

    def process_input(self, text: str, markers: Optional[Dict] = None) -> ProcessedInput:
        """
        Process raw text input and optional marker files
        """
        return ProcessedInput(
            text=self.preprocessor.clean(text),
            language=self.detect_language(text),
            markers=self.load_markers(markers),
            metadata=self.extract_metadata(text)
        )
```

#### 1.2 Marker Manager
```python
class MarkerManager:
    def __init__(self, marker_schema_path: str):
        self.schema = self.load_schema(marker_schema_path)
        self.validator = MarkerValidator(self.schema)

    def load_markers(self, marker_file: str) -> List[Marker]:
        """Load and validate marker files (UB_markers_canonical.ld35.json format)"""
        pass

    def validate_markers(self, markers: List[Marker]) -> ValidationResult:
        """Validate markers against schema and bias taxonomy"""
        pass
```

### 2. Bias Engine Architecture

#### 2.1 Core Detection Pipeline
```python
class BiasDetectionEngine:
    def __init__(self, config: BiasConfig):
        self.nlp_spacy = spacy.load("en_core_web_lg")
        self.nlp_stanza = stanza.Pipeline("en")
        self.transformer_model = AutoModel.from_pretrained("distilbert-base-multilingual-cased")
        self.rule_engine = RuleBasedDetector(config.rules_path)
        self.taxonomy = BiasTaxonomy.load(config.taxonomy_path)

    def detect_bias(self, text: ProcessedInput) -> List[BiasDetection]:
        """
        Multi-modal bias detection combining:
        - Rule-based pattern matching
        - Named entity recognition (spaCy)
        - Dependency parsing (Stanza)
        - Contextual embeddings (Transformers)
        """
        detections = []

        # Rule-based detection
        rule_detections = self.rule_engine.detect(text)

        # NLP-based detection
        nlp_detections = self.nlp_detect(text)

        # Transformer-based detection
        ml_detections = self.ml_detect(text)

        # Aggregate and resolve conflicts
        return self.aggregate_detections(rule_detections, nlp_detections, ml_detections)
```

#### 2.2 Intersectional Taxonomy Handler
```python
class BiasTaxonomy:
    FAMILIES = {
        "racism": ["stereotyping", "dehumanization", "othering"],
        "sexism": ["objectification", "victim_blaming", "gaslighting"],
        "classism": ["economic_shaming", "privilege_denial"],
        "ableism": ["medical_model", "inspiration_porn"],
        "ageism": ["patronization", "dismissal"],
        "queerphobia": ["heteronormativity", "binary_enforcement"],
        "xenophobia": ["cultural_erasure", "assimilation_pressure"],
        "religious": ["theological_supremacy", "secular_bias"],
        "intersectional": ["compound_marginalization", "identity_hierarchy"]
    }

    def classify_bias(self, span: TextSpan, features: Dict) -> BiasClassification:
        """Classify bias span into family/subtype with confidence score"""
        pass
```

### 3. Cultural Engine Architecture

#### 3.1 Cultural Profile Manager
```python
class CulturalProfileManager:
    def __init__(self):
        self.hofstede_profiles = self.load_hofstede_profiles()
        self.globe_profiles = self.load_globe_profiles()
        self.hall_profiles = self.load_hall_profiles()

    def get_cultural_context(self, sender_culture: str, receiver_culture: str) -> CulturalContext:
        """
        Generate cultural context analysis based on multiple models
        """
        return CulturalContext(
            sender=self.get_profile(sender_culture),
            receiver=self.get_profile(receiver_culture),
            differences=self.calculate_differences(sender_culture, receiver_culture),
            severity_adjustments=self.calculate_severity_adjustments()
        )
```

#### 3.2 Severity Adjustment Engine
```python
class SeverityAdjustmentEngine:
    def adjust_severity(self, bias_detection: BiasDetection, cultural_context: CulturalContext) -> AdjustedSeverity:
        """
        Adjust bias severity based on cultural context using Hofstede dimensions:
        - Power Distance (PDI)
        - Individualism vs Collectivism (IDV)
        - Masculinity vs Femininity (MAS)
        - Uncertainty Avoidance (UAI)
        - Long-term Orientation (LTO)
        - Indulgence vs Restraint (IVR)
        """
        base_severity = bias_detection.severity

        # Cultural adjustment factors
        pdi_factor = self.calculate_pdi_impact(bias_detection, cultural_context)
        idv_factor = self.calculate_idv_impact(bias_detection, cultural_context)
        mas_factor = self.calculate_mas_impact(bias_detection, cultural_context)
        uai_factor = self.calculate_uai_impact(bias_detection, cultural_context)

        # Compound adjustment
        adjusted_severity = base_severity * pdi_factor * idv_factor * mas_factor * uai_factor

        return AdjustedSeverity(
            original=base_severity,
            adjusted=min(10.0, max(0.0, adjusted_severity)),
            explanation=self.generate_explanation(cultural_context, bias_detection),
            confidence=self.calculate_confidence()
        )
```

### 4. LLM Integration Architecture

#### 4.1 Prompt Template Engine
```python
class PromptTemplateEngine:
    def __init__(self, template_config: Dict):
        self.templates = self.load_templates(template_config)
        self.jinja_env = jinja2.Environment()

    def generate_debias_prompt(self, bias_detection: BiasDetection, cultural_context: CulturalContext) -> str:
        """
        Generate contextual prompt for LLM debiasing based on:
        - Bias family and subtype
        - Cultural sender/receiver context
        - Severity level
        - Required output format (A/B variants)
        """
        template = self.templates[f"debias_{bias_detection.family}"]

        return template.render(
            bias_span=bias_detection.span,
            bias_family=bias_detection.family,
            bias_subtype=bias_detection.subtype,
            severity=bias_detection.adjusted_severity,
            sender_culture=cultural_context.sender.name,
            receiver_culture=cultural_context.receiver.name,
            cultural_explanation=cultural_context.explanation,
            output_language=cultural_context.receiver.language
        )
```

#### 4.2 LLM Client Manager
```python
class LLMClientManager:
    def __init__(self, config: LLMConfig):
        self.clients = {
            "openai": OpenAIClient(config.openai),
            "anthropic": AnthropicClient(config.anthropic),
            "azure": AzureOpenAIClient(config.azure)
        }
        self.fallback_chain = config.fallback_chain
        self.rate_limiter = RateLimiter(config.rate_limits)

    async def process_debiasing(self, prompt: str, provider: str = "primary") -> LLMResponse:
        """
        Process debiasing request with fallback support
        """
        for client_name in self.fallback_chain:
            try:
                async with self.rate_limiter.acquire(client_name):
                    response = await self.clients[client_name].complete(prompt)
                    if self.validate_response(response):
                        return response
            except Exception as e:
                logger.warning(f"Client {client_name} failed: {e}")
                continue

        raise LLMProcessingError("All LLM providers failed")
```

### 5. Self-Bias Check Architecture

#### 5.1 Epistemological Classifier
```python
class EpistemologicalClassifier:
    CLAIM_TYPES = {
        "factual": ["Faktisch korrekt", "Factually correct"],
        "logical": ["Logisch scheint mir", "Logically, it appears"],
        "subjective": ["Rein subjektiv", "Purely subjectively"]
    }

    def classify_claims(self, text: str) -> List[ClassifiedClaim]:
        """
        Classify each claim in the text as factual/logical/subjective
        and add appropriate epistemic prefixes
        """
        sentences = self.sentence_tokenizer.tokenize(text)
        classified_claims = []

        for sentence in sentences:
            claim_type = self.classify_sentence(sentence)
            prefix = self.get_prefix(claim_type, self.detect_language(sentence))

            classified_claims.append(ClassifiedClaim(
                original=sentence,
                type=claim_type,
                prefixed=f"{prefix}, {sentence.lower()}",
                confidence=self.get_classification_confidence(sentence)
            ))

        return classified_claims
```

#### 5.2 Overconfidence Detector
```python
class OverconfidenceDetector:
    CONFIDENCE_MARKERS = {
        "high": ["definitely", "certainly", "absolutely", "without doubt"],
        "medium": ["probably", "likely", "appears to be"],
        "hedged": ["might", "could", "possibly", "potentially"]
    }

    def reduce_overconfidence(self, text: str) -> str:
        """
        Detect and reduce overconfident language in LLM outputs
        """
        confidence_level = self.detect_confidence_level(text)

        if confidence_level == "high":
            return self.add_hedging_language(text)

        return text
```

### 6. Dashboard Architecture

#### 6.1 React Component Structure
```typescript
// Core Dashboard Components
interface DashboardProps {
    analysisResult: BiasAnalysisResult;
    culturalContext: CulturalContext;
    onExport: (format: ExportFormat) => void;
}

const BiazDashboard: React.FC<DashboardProps> = ({
    analysisResult,
    culturalContext,
    onExport
}) => {
    return (
        <div className="bias-dashboard">
            <BiasHeatmap detections={analysisResult.detections} />
            <MarkerExplorer markers={analysisResult.markers} />
            <SideBySideComparison
                original={analysisResult.original}
                variantA={analysisResult.variantA}
                variantB={analysisResult.variantB}
            />
            <SeverityTrendChart trends={analysisResult.severityTrends} />
            <IntersectionalOverlapMatrix overlaps={analysisResult.overlaps} />
            <CulturalContextPanel context={culturalContext} />
            <HofstedeRadarChart profiles={culturalContext.profiles} />
            <SelfBiasCheckRibbon checks={analysisResult.selfBiasChecks} />
        </div>
    );
};
```

#### 6.2 Real-time Data Management
```typescript
class BiasAnalysisService {
    private apiClient: ApiClient;
    private eventBus: EventBus;

    async analyzeText(input: AnalysisInput): Promise<AnalysisResult> {
        this.eventBus.emit('analysis:started', input);

        try {
            const result = await this.apiClient.post('/analyze/full', input);
            this.eventBus.emit('analysis:completed', result);
            return result;
        } catch (error) {
            this.eventBus.emit('analysis:failed', error);
            throw error;
        }
    }

    subscribeToProgress(callback: (progress: AnalysisProgress) => void): void {
        this.eventBus.on('analysis:progress', callback);
    }
}
```

## 🚀 API Specification

### Core Endpoints

#### POST /api/v1/analyze/full
```json
{
    "text": "Input text to analyze",
    "sender_culture": "de",
    "receiver_culture": "jp",
    "context": "political_discussion",
    "formality_level": "neutral",
    "markers_file": "optional_markers.json",
    "options": {
        "include_variants": true,
        "generate_markers": true,
        "cultural_adaptation": true,
        "self_bias_check": true
    }
}
```

#### Response Schema
```json
{
    "analysis_id": "uuid",
    "timestamp": "2024-12-02T18:00:00Z",
    "input": {
        "text": "Original text",
        "language": "de",
        "sender_culture": "de",
        "receiver_culture": "jp"
    },
    "detections": [
        {
            "span": "problematic text",
            "start_pos": 15,
            "end_pos": 30,
            "bias_family": "racism",
            "bias_subtype": "stereotyping",
            "severity_raw": 8.5,
            "severity_adjusted": {
                "sender": 7.9,
                "receiver": 9.2
            },
            "confidence": 0.87,
            "explanation": "Cultural explanation..."
        }
    ],
    "variants": {
        "variant_a": "Neutral rewrite",
        "variant_b": "Emotional but debiased rewrite"
    },
    "cultural_analysis": {
        "sender_profile": {...},
        "receiver_profile": {...},
        "differences": {...},
        "recommendations": [...]
    },
    "markers": {
        "neutralized": [...],
        "generated": [...]
    },
    "self_bias_check": {
        "claims": [...],
        "confidence_adjustments": [...],
        "epistemic_classification": [...]
    },
    "summary": "Narrative summary with epistemic prefixes...",
    "metadata": {
        "processing_time_ms": 2341,
        "llm_calls": 3,
        "confidence_score": 0.91
    }
}
```

## 🔒 Security & Privacy

### Data Protection
- No persistent storage of input texts
- Hash-based logging for privacy
- Anonymization before LLM processing
- GDPR-compliant data handling

### API Security
- Rate limiting per client
- Input validation and sanitization
- Output content filtering
- Audit logging for compliance

## 📊 Performance Targets

### Response Time
- Median: <5 seconds
- 95th percentile: <10 seconds
- Timeout: 30 seconds

### Throughput
- Single analysis: 100 req/min
- Batch analysis: 10 batches/min
- Concurrent users: 50

### Resource Usage
- Memory: <4GB per instance
- CPU: <80% utilization
- Storage: Minimal (cache only)

## 🏗️ Deployment Architecture

```yaml
# docker-compose.yml
version: '3.8'
services:
  bias-api:
    build: ./bias-engine
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./config:/app/config
      - ./models:/app/models

  bias-dashboard:
    build: ./bias-dashboard
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://bias-api:8000

  redis-cache:
    image: redis:alpine
    ports:
      - "6379:6379"

  monitoring:
    image: grafana/grafana
    ports:
      - "3001:3000"
```

This technical architecture provides a robust, scalable foundation for the BiazNeutralize AI system while ensuring maintainability and extensibility for future enhancements.