"""
LLM Integration module for bias detection and neutralization.

This module provides comprehensive LLM integration capabilities including:
- Multi-provider LLM client support (OpenAI, Anthropic, Azure)
- Prompt management with YAML templates
- Debiasing pipeline with A/B variants
- Self-bias check system with epistemic classification
- Cultural context integration
- Rate limiting and retry mechanisms
"""

from .client import LLMClient, LLMProvider
from .prompts import PromptManager
from .pipeline import DebiasingPipeline
from .self_bias import SelfBiasChecker
from .models import (
    LLMResponse,
    DebiasingRequest,
    DebiasingResponse,
    BatchDebiasingRequest,
    BatchDebiasingResponse,
    MarkerGenerationRequest,
    MarkerGenerationResponse,
    SelfBiasCheckResponse,
)

__all__ = [
    "LLMClient",
    "LLMProvider",
    "PromptManager",
    "DebiasingPipeline",
    "SelfBiasChecker",
    "LLMResponse",
    "DebiasingRequest",
    "DebiasingResponse",
    "BatchDebiasingRequest",
    "BatchDebiasingResponse",
    "MarkerGenerationRequest",
    "MarkerGenerationResponse",
    "SelfBiasCheckResponse",
]
