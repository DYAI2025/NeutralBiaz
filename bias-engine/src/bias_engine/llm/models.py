"""
Pydantic models for LLM integration and debiasing operations.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from enum import Enum

from pydantic import BaseModel, Field, validator


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    GOOGLE = "google"


class EpistemicClassification(str, Enum):
    """Epistemic classification for self-bias checking."""
    FAKTISCH = "faktisch"  # factual/objective
    LOGISCH = "logisch"   # logical/rational
    SUBJEKTIV = "subjektiv"  # subjective/opinion


class CulturalSeverityLevel(BaseModel):
    """Cultural severity adjustment for bias spans."""
    sender_culture: str = Field(description="Sender culture code")
    receiver_culture: str = Field(description="Receiver culture code")
    raw_severity: float = Field(ge=0, le=10, description="Raw severity score")
    sender_severity: float = Field(ge=0, le=10, description="Sender-adjusted severity")
    receiver_severity: float = Field(ge=0, le=10, description="Receiver-adjusted severity")
    cultural_explanation: str = Field(description="Cultural adjustment explanation")


class BiasSpan(BaseModel):
    """Individual bias span for debiasing."""
    span_id: str = Field(description="Unique span identifier")
    full_sentence_or_paragraph: str = Field(description="Complete context containing the bias")
    bias_span: str = Field(description="Specific problematic text span")
    bias_family: str = Field(description="Bias category family")
    bias_subtype: str = Field(description="Specific bias subtype")
    severity: CulturalSeverityLevel = Field(description="Cultural severity adjustments")
    start_position: int = Field(ge=0, description="Start position in original text")
    end_position: int = Field(ge=0, description="End position in original text")


class LLMResponse(BaseModel):
    """Base LLM response model."""
    content: str = Field(description="Response content")
    provider: LLMProvider = Field(description="LLM provider used")
    model: str = Field(description="Specific model used")
    tokens_used: Dict[str, int] = Field(description="Token usage statistics")
    response_time: float = Field(ge=0, description="Response time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DebiasingRequest(BaseModel):
    """Request for debiasing a single span."""
    bias_span: BiasSpan = Field(description="Bias span to neutralize")
    input_language: str = Field(description="Original text language")
    output_language: str = Field(description="Target language for output")
    sender_culture: str = Field(description="Sender culture context")
    receiver_culture: str = Field(description="Receiver culture context")
    context_topic: str = Field(description="Context or topic of the text")
    audience: str = Field(description="Target audience or setting")
    formality_level: str = Field(description="Required formality level")

    @validator("input_language", "output_language")
    def validate_language_codes(cls, v):
        """Validate language codes are properly formatted."""
        if len(v) < 2:
            raise ValueError("Language codes must be at least 2 characters")
        return v.lower()


class DebiasingResponse(BaseModel):
    """Response for single span debiasing."""
    span_id: str = Field(description="Original span identifier")
    language: str = Field(description="Response language")
    bias_family: str = Field(description="Detected bias family")
    bias_subtype: str = Field(description="Detected bias subtype")
    analysis_explanation: str = Field(description="Why the span is problematic")
    can_preserve_core_intent: bool = Field(description="Whether core intent can be preserved")
    variant_a_rewrite: str = Field(description="Neutral, factual alternative")
    variant_b_rewrite: str = Field(description="Emotionally similar but bias-reduced alternative")
    safety_notes: str = Field(description="Additional safety considerations")
    confidence_score: float = Field(ge=0, le=1, description="Confidence in the rewrite")
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchDebiasingRequest(BaseModel):
    """Request for batch debiasing multiple spans."""
    spans: List[BiasSpan] = Field(min_items=1, max_items=50, description="Bias spans to process")
    full_document_text: str = Field(description="Complete document for context")
    input_language: str = Field(description="Document language")
    output_language: str = Field(description="Target language for output")
    sender_culture: str = Field(description="Sender culture context")
    receiver_culture: str = Field(description="Receiver culture context")
    context_topic: str = Field(description="Document topic")
    audience: str = Field(description="Target audience")
    formality_level: str = Field(description="Required formality level")

    @validator("spans")
    def validate_unique_span_ids(cls, v):
        """Ensure all span IDs are unique."""
        span_ids = [span.span_id for span in v]
        if len(span_ids) != len(set(span_ids)):
            raise ValueError("All span IDs must be unique")
        return v


class BatchDebiasingResponse(BaseModel):
    """Response for batch debiasing."""
    language: str = Field(description="Response language")
    spans: List[DebiasingResponse] = Field(description="Individual span responses")
    total_processed: int = Field(ge=0, description="Number of spans processed")
    processing_time: float = Field(ge=0, description="Total processing time")
    batch_metadata: Dict[str, Any] = Field(default_factory=dict)


class MarkerDefinition(BaseModel):
    """Definition of a bias detection marker."""
    id: str = Field(description="Marker identifier")
    name: str = Field(description="Human-readable marker name")
    description: str = Field(description="Marker description")
    rationale: str = Field(description="Why this marker is important")
    positive_examples: List[str] = Field(min_items=3, description="Examples that match this marker")
    counter_example: str = Field(description="Example that should NOT match")
    severity_hint: str = Field(description="Suggested severity range")
    languages: List[str] = Field(description="Supported languages")


class MarkerGenerationRequest(BaseModel):
    """Request for generating new bias markers."""
    bias_family: str = Field(description="Bias category family")
    bias_subtype: str = Field(description="Specific bias subtype")
    bias_description: str = Field(description="Description of the bias pattern")
    output_language: str = Field(description="Target language")
    domain: str = Field(description="Application domain")
    primary_cultures: List[str] = Field(description="Primary cultural contexts")
    old_markers: Optional[List[Dict[str, Any]]] = Field(default=None, description="Existing markers")


class MarkerGenerationResponse(BaseModel):
    """Response for marker generation."""
    bias_family: str = Field(description="Bias category family")
    bias_subtype: str = Field(description="Bias subtype")
    language: str = Field(description="Generated language")
    markers: List[MarkerDefinition] = Field(description="Generated markers")
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)


class SelfBiasCheckRequest(BaseModel):
    """Request for self-bias checking."""
    text: str = Field(description="Text to check for self-bias")
    context: str = Field(description="Context of the statement")
    claimed_classification: Optional[EpistemicClassification] = Field(
        default=None,
        description="Claimed epistemic classification"
    )


class SelfBiasCheckResponse(BaseModel):
    """Response for self-bias checking."""
    original_text: str = Field(description="Original text checked")
    epistemic_classification: EpistemicClassification = Field(description="Determined classification")
    overconfidence_detected: bool = Field(description="Whether overconfidence was detected")
    bias_indicators: List[str] = Field(description="Detected bias indicators")
    corrected_text: str = Field(description="Bias-corrected version with prefix")
    confidence_score: float = Field(ge=0, le=1, description="Confidence in the assessment")
    explanation: str = Field(description="Explanation of the bias check")


class LLMConfig(BaseModel):
    """LLM client configuration."""
    provider: LLMProvider = Field(description="LLM provider")
    api_key: str = Field(description="API key for the provider")
    model: str = Field(description="Model name")
    base_url: Optional[str] = Field(default=None, description="Custom base URL")
    max_tokens: int = Field(default=4096, description="Maximum tokens per request")
    temperature: float = Field(default=0.1, ge=0, le=2, description="Sampling temperature")
    timeout: int = Field(default=60, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries")


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: int = Field(default=60, description="Requests per minute limit")
    tokens_per_minute: int = Field(default=100000, description="Tokens per minute limit")
    burst_size: int = Field(default=10, description="Burst request allowance")


class PromptTemplate(BaseModel):
    """Prompt template configuration."""
    name: str = Field(description="Template name")
    role: str = Field(description="Message role (system/user)")
    content: str = Field(description="Template content with variables")
    variables: List[str] = Field(description="Required template variables")
    description: Optional[str] = Field(default=None, description="Template description")