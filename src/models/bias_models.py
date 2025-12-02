#!/usr/bin/env python3
"""
Bias Detection Models

Data models for representing bias families, subtypes, detection results, and spans.
Provides comprehensive type safety and validation for the bias detection system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple, Any
from enum import Enum
import json
from datetime import datetime
import uuid


class BiasConfidenceLevel(Enum):
    """Confidence levels for bias detection"""
    UNCERTAIN = "uncertain"
    LIKELY = "likely"
    CONFIDENT = "confident"
    CERTAIN = "certain"


class BiasSeverityLevel(Enum):
    """Severity levels for bias detection"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DetectionMethod(Enum):
    """Methods used for bias detection"""
    RULE_BASED = "rule_based"
    ML_CLASSIFICATION = "ml_classification"
    TRANSFORMER_MODEL = "transformer_model"
    HYBRID = "hybrid"
    INTERSECTIONAL = "intersectional"


@dataclass
class BiasSubtype:
    """Represents a specific subtype of bias within a family"""
    id: str
    name: str
    description: str
    severity_multiplier: float
    patterns: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    regex_patterns: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not 0.0 <= self.severity_multiplier <= 2.0:
            raise ValueError(f"Severity multiplier must be between 0.0 and 2.0, got {self.severity_multiplier}")
        if not self.id or not self.name:
            raise ValueError("ID and name are required for BiasSubtype")


@dataclass
class BiasFamily:
    """Represents a family of related bias types"""
    id: str
    name: str
    description: str
    weight: float
    subtypes: Dict[str, BiasSubtype] = field(default_factory=dict)
    intersectional_weights: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0.0 <= self.weight <= 2.0:
            raise ValueError(f"Weight must be between 0.0 and 2.0, got {self.weight}")
        if not self.id or not self.name:
            raise ValueError("ID and name are required for BiasFamily")
        
        # Convert dict subtypes to BiasSubtype objects if needed
        for key, value in self.subtypes.items():
            if isinstance(value, dict):
                self.subtypes[key] = BiasSubtype(**value)
    
    def get_subtype(self, subtype_id: str) -> Optional[BiasSubtype]:
        """Get a specific subtype by ID"""
        return self.subtypes.get(subtype_id)
    
    def calculate_base_severity(self, subtype_id: str) -> float:
        """Calculate base severity for a subtype"""
        subtype = self.get_subtype(subtype_id)
        if not subtype:
            return 0.5  # Default moderate severity
        return self.weight * subtype.severity_multiplier


@dataclass
class BiasSpan:
    """Represents a span of text containing detected bias"""
    start: int
    end: int
    text: str
    bias_family: str
    bias_subtype: str
    severity: float
    confidence: float
    method: DetectionMethod
    explanation: str = ""
    context_window: str = ""
    intersectional_factors: List[str] = field(default_factory=list)
    cultural_modifiers: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0 <= self.start < self.end:
            raise ValueError(f"Invalid span coordinates: start={self.start}, end={self.end}")
        if not 0.0 <= self.severity <= 10.0:
            raise ValueError(f"Severity must be between 0.0 and 10.0, got {self.severity}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if self.end - self.start != len(self.text):
            raise ValueError("Span length doesn't match text length")
    
    @property
    def severity_level(self) -> BiasSeverityLevel:
        """Convert numeric severity to categorical level"""
        if self.severity < 3.0:
            return BiasSeverityLevel.LOW
        elif self.severity < 6.0:
            return BiasSeverityLevel.MEDIUM
        elif self.severity < 8.0:
            return BiasSeverityLevel.HIGH
        else:
            return BiasSeverityLevel.CRITICAL
    
    @property
    def confidence_level(self) -> BiasConfidenceLevel:
        """Convert numeric confidence to categorical level"""
        if self.confidence < 0.4:
            return BiasConfidenceLevel.UNCERTAIN
        elif self.confidence < 0.6:
            return BiasConfidenceLevel.LIKELY
        elif self.confidence < 0.8:
            return BiasConfidenceLevel.CONFIDENT
        else:
            return BiasConfidenceLevel.CERTAIN
    
    def overlaps_with(self, other: 'BiasSpan') -> bool:
        """Check if this span overlaps with another span"""
        return not (self.end <= other.start or other.end <= self.start)
    
    def merge_with(self, other: 'BiasSpan') -> 'BiasSpan':
        """Merge this span with another overlapping span"""
        if not self.overlaps_with(other):
            raise ValueError("Cannot merge non-overlapping spans")
        
        new_start = min(self.start, other.start)
        new_end = max(self.end, other.end)
        new_text = self.context_window[new_start:new_end] if self.context_window else ""
        
        # Use higher severity and confidence
        new_severity = max(self.severity, other.severity)
        new_confidence = max(self.confidence, other.confidence)
        
        # Combine intersectional factors
        combined_factors = list(set(self.intersectional_factors + other.intersectional_factors))
        
        return BiasSpan(
            start=new_start,
            end=new_end,
            text=new_text,
            bias_family=self.bias_family if self.confidence >= other.confidence else other.bias_family,
            bias_subtype=self.bias_subtype if self.confidence >= other.confidence else other.bias_subtype,
            severity=new_severity,
            confidence=new_confidence,
            method=DetectionMethod.HYBRID,
            explanation=f"Merged: {self.explanation} | {other.explanation}",
            context_window=self.context_window or other.context_window,
            intersectional_factors=combined_factors,
            cultural_modifiers={**self.cultural_modifiers, **other.cultural_modifiers}
        )


@dataclass
class IntersectionalAnalysis:
    """Analysis of intersectional bias patterns"""
    detected_identities: List[str]
    intersection_score: float
    amplification_factor: float
    erasure_indicators: List[str] = field(default_factory=list)
    privilege_indicators: List[str] = field(default_factory=list)
    marginalization_indicators: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not 0.0 <= self.intersection_score <= 1.0:
            raise ValueError(f"Intersection score must be between 0.0 and 1.0, got {self.intersection_score}")
        if not 1.0 <= self.amplification_factor <= 5.0:
            raise ValueError(f"Amplification factor must be between 1.0 and 5.0, got {self.amplification_factor}")


@dataclass
class BiasDetectionResult:
    """Complete result of bias detection analysis"""
    id: str
    text: str
    language: str
    detected_spans: List[BiasSpan]
    overall_severity: float
    overall_confidence: float
    intersectional_analysis: Optional[IntersectionalAnalysis] = None
    cultural_context: Dict[str, Any] = field(default_factory=dict)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    model_versions: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not 0.0 <= self.overall_severity <= 10.0:
            raise ValueError(f"Overall severity must be between 0.0 and 10.0, got {self.overall_severity}")
        if not 0.0 <= self.overall_confidence <= 1.0:
            raise ValueError(f"Overall confidence must be between 0.0 and 1.0, got {self.overall_confidence}")
    
    @property
    def bias_families_detected(self) -> List[str]:
        """Get list of unique bias families detected"""
        return list(set(span.bias_family for span in self.detected_spans))
    
    @property
    def bias_subtypes_detected(self) -> List[str]:
        """Get list of unique bias subtypes detected"""
        return list(set(f"{span.bias_family}.{span.bias_subtype}" for span in self.detected_spans))
    
    @property
    def has_intersectional_bias(self) -> bool:
        """Check if intersectional bias was detected"""
        return (
            self.intersectional_analysis is not None and 
            len(self.intersectional_analysis.detected_identities) > 1
        )
    
    def get_spans_by_family(self, family: str) -> List[BiasSpan]:
        """Get all spans for a specific bias family"""
        return [span for span in self.detected_spans if span.bias_family == family]
    
    def get_spans_by_severity(self, min_severity: float) -> List[BiasSpan]:
        """Get all spans above a minimum severity threshold"""
        return [span for span in self.detected_spans if span.severity >= min_severity]
    
    def get_spans_by_confidence(self, min_confidence: float) -> List[BiasSpan]:
        """Get all spans above a minimum confidence threshold"""
        return [span for span in self.detected_spans if span.confidence >= min_confidence]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "text": self.text,
            "language": self.language,
            "detected_spans": [
                {
                    "start": span.start,
                    "end": span.end,
                    "text": span.text,
                    "bias_family": span.bias_family,
                    "bias_subtype": span.bias_subtype,
                    "severity": span.severity,
                    "confidence": span.confidence,
                    "method": span.method.value,
                    "explanation": span.explanation,
                    "severity_level": span.severity_level.value,
                    "confidence_level": span.confidence_level.value,
                    "intersectional_factors": span.intersectional_factors,
                    "cultural_modifiers": span.cultural_modifiers
                }
                for span in self.detected_spans
            ],
            "overall_severity": self.overall_severity,
            "overall_confidence": self.overall_confidence,
            "bias_families_detected": self.bias_families_detected,
            "bias_subtypes_detected": self.bias_subtypes_detected,
            "has_intersectional_bias": self.has_intersectional_bias,
            "intersectional_analysis": {
                "detected_identities": self.intersectional_analysis.detected_identities,
                "intersection_score": self.intersectional_analysis.intersection_score,
                "amplification_factor": self.intersectional_analysis.amplification_factor,
                "erasure_indicators": self.intersectional_analysis.erasure_indicators,
                "privilege_indicators": self.intersectional_analysis.privilege_indicators,
                "marginalization_indicators": self.intersectional_analysis.marginalization_indicators
            } if self.intersectional_analysis else None,
            "cultural_context": self.cultural_context,
            "processing_metadata": self.processing_metadata,
            "timestamp": self.timestamp.isoformat(),
            "model_versions": self.model_versions
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class BiasClassification:
    """Result of bias type classification"""
    bias_family: str
    bias_subtype: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    alternative_classifications: List[Tuple[str, str, float]] = field(default_factory=list)
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
    
    @property
    def full_type(self) -> str:
        """Get full bias type as family.subtype"""
        return f"{self.bias_family}.{self.bias_subtype}"


@dataclass
class BiasPatternMatch:
    """Result of pattern-based bias detection"""
    pattern: str
    match_text: str
    start: int
    end: int
    pattern_type: str  # 'keyword', 'regex', 'semantic'
    confidence: float
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if not 0 <= self.start < self.end:
            raise ValueError(f"Invalid match coordinates: start={self.start}, end={self.end}")


# Validation functions
def validate_bias_family_config(config: Dict[str, Any]) -> bool:
    """Validate bias family configuration structure"""
    required_fields = ['id', 'name', 'description', 'weight']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    if not isinstance(config['weight'], (int, float)) or not 0.0 <= config['weight'] <= 2.0:
        raise ValueError(f"Weight must be a number between 0.0 and 2.0, got {config['weight']}")
    
    if 'subtypes' in config:
        for subtype_id, subtype_config in config['subtypes'].items():
            validate_bias_subtype_config(subtype_config)
    
    return True


def validate_bias_subtype_config(config: Dict[str, Any]) -> bool:
    """Validate bias subtype configuration structure"""
    required_fields = ['id', 'name', 'description', 'severity_multiplier']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in subtype: {field}")
    
    if not isinstance(config['severity_multiplier'], (int, float)) or not 0.0 <= config['severity_multiplier'] <= 2.0:
        raise ValueError(f"Severity multiplier must be between 0.0 and 2.0, got {config['severity_multiplier']}")
    
    return True
