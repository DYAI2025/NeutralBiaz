#!/usr/bin/env python3
"""
Core Bias Detection Engine

Main interface for bias detection combining:
- Rule-based detection
- ML-based classification
- Intersectional analysis
- Confidence and severity scoring
- Error handling and logging
"""

import logging
import traceback
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import uuid

from models.bias_models import (
    BiasSpan, BiasDetectionResult, BiasClassification,
    IntersectionalAnalysis, DetectionMethod
)
from .rule_based_detector import get_rule_based_detector
from .ml_classifier import get_ml_classifier
from .scoring_algorithms import (
    get_confidence_calculator, get_severity_calculator, get_aggregated_scoring,
    DetectionSignal, ConfidenceAggregationMethod, SeverityCalculationMethod
)
from .taxonomy_loader import get_taxonomy_loader
from .nlp_pipeline import get_nlp_pipeline


logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    """Configuration for bias detection"""
    enable_rule_based: bool = True
    enable_ml_classification: bool = True
    enable_intersectional_analysis: bool = True
    confidence_threshold: float = 0.3
    severity_threshold: float = 2.0
    max_spans_per_text: int = 50
    confidence_method: ConfidenceAggregationMethod = ConfidenceAggregationMethod.ENSEMBLE
    severity_method: SeverityCalculationMethod = SeverityCalculationMethod.INTERSECTIONAL
    merge_overlapping_spans: bool = True
    include_low_confidence: bool = False
    cultural_adaptation: bool = False
    language_auto_detect: bool = True


class IntersectionalAnalyzer:
    """Analyzes intersectional bias patterns"""
    
    def __init__(self):
        self.taxonomy = get_taxonomy_loader()
        self.nlp = get_nlp_pipeline()
    
    def analyze_intersectional_bias(self, text: str, detected_spans: List[BiasSpan]) -> Optional[IntersectionalAnalysis]:
        """Analyze intersectional bias patterns in detected spans"""
        if not detected_spans or len(detected_spans) < 2:
            return None
        
        try:
            # Extract detected identity categories
            detected_identities = self._extract_identity_categories(detected_spans)
            
            if len(detected_identities) < 2:
                return None
            
            # Calculate intersection score
            intersection_score = self._calculate_intersection_score(detected_spans)
            
            # Calculate amplification factor
            amplification_factor = self.taxonomy.calculate_intersectional_amplification(
                detected_identities
            )
            
            # Analyze for different types of intersectional bias
            erasure_indicators = self._detect_erasure_indicators(text, detected_identities)
            privilege_indicators = self._detect_privilege_indicators(text, detected_identities)
            marginalization_indicators = self._detect_marginalization_indicators(text, detected_identities)
            
            return IntersectionalAnalysis(
                detected_identities=detected_identities,
                intersection_score=intersection_score,
                amplification_factor=amplification_factor,
                erasure_indicators=erasure_indicators,
                privilege_indicators=privilege_indicators,
                marginalization_indicators=marginalization_indicators
            )
            
        except Exception as e:
            logger.error(f"Error in intersectional analysis: {e}")
            return None
    
    def _extract_identity_categories(self, spans: List[BiasSpan]) -> List[str]:
        """Extract unique identity categories from bias spans"""
        identities = set()
        
        for span in spans:
            # Map bias types to identity categories
            bias_type = f"{span.bias_family}.{span.bias_subtype}"
            
            # Mapping bias types to identity categories
            if span.bias_family == "demographic":
                if span.bias_subtype in ["racial", "ethnic"]:
                    identities.add("race_ethnicity")
                elif span.bias_subtype == "gender":
                    identities.add("gender")
                elif span.bias_subtype == "age":
                    identities.add("age")
                elif span.bias_subtype == "sexual_orientation":
                    identities.add("sexual_orientation")
            elif span.bias_family == "socioeconomic":
                identities.add("class")
            elif span.bias_family == "cultural":
                if span.bias_subtype == "religious":
                    identities.add("religion")
                elif span.bias_subtype == "nationality":
                    identities.add("nationality")
            elif span.bias_family == "physical":
                if span.bias_subtype == "disability":
                    identities.add("disability")
        
        return list(identities)
    
    def _calculate_intersection_score(self, spans: List[BiasSpan]) -> float:
        """Calculate how much the detected biases intersect"""
        if len(spans) < 2:
            return 0.0
        
        # Calculate span overlap
        total_overlap = 0.0
        span_pairs = 0
        
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                span_pairs += 1
                
                # Check for overlap
                if spans[i].overlaps_with(spans[j]):
                    # Calculate overlap ratio
                    overlap_start = max(spans[i].start, spans[j].start)
                    overlap_end = min(spans[i].end, spans[j].end)
                    overlap_length = overlap_end - overlap_start
                    
                    min_span_length = min(spans[i].end - spans[i].start, spans[j].end - spans[j].start)
                    overlap_ratio = overlap_length / min_span_length if min_span_length > 0 else 0
                    
                    total_overlap += overlap_ratio
        
        return total_overlap / span_pairs if span_pairs > 0 else 0.0
    
    def _detect_erasure_indicators(self, text: str, identities: List[str]) -> List[str]:
        """Detect indicators of identity erasure"""
        indicators = []
        text_lower = text.lower()
        
        # Common erasure patterns
        erasure_patterns = {
            "color_blind": ["color blind", "colour blind", "don't see color", "don't see colour"],
            "gender_blind": ["gender doesn't matter", "gender blind"],
            "universal": ["we're all the same", "we're all human", "there's only one race"],
            "post_racial": ["post-racial", "beyond race", "race doesn't exist"]
        }
        
        for pattern_type, patterns in erasure_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    indicators.append(pattern_type)
                    break
        
        return indicators
    
    def _detect_privilege_indicators(self, text: str, identities: List[str]) -> List[str]:
        """Detect indicators of privilege dynamics"""
        indicators = []
        text_lower = text.lower()
        
        # Privilege-related patterns
        privilege_patterns = {
            "merit_myth": ["work harder", "pull yourself up", "bootstrap", "earned everything"],
            "reverse_discrimination": ["reverse racism", "reverse discrimination", "anti-white"],
            "privilege_denial": ["no privilege", "everyone has equal opportunity"],
            "victim_blaming": ["playing victim", "victim mentality", "just complaining"]
        }
        
        for pattern_type, patterns in privilege_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    indicators.append(pattern_type)
                    break
        
        return indicators
    
    def _detect_marginalization_indicators(self, text: str, identities: List[str]) -> List[str]:
        """Detect indicators of marginalization"""
        indicators = []
        text_lower = text.lower()
        
        # Marginalization patterns
        marginalization_patterns = {
            "exclusion": ["not welcome", "don't belong", "go back", "not from here"],
            "devaluation": ["less than", "inferior", "not as good", "second class"],
            "stereotyping": ["typical", "all of them", "their kind", "what you expect"],
            "tokenism": ["diversity hire", "quota", "token", "affirmative action"]
        }
        
        for pattern_type, patterns in marginalization_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    indicators.append(pattern_type)
                    break
        
        return indicators


class BiasDetectionEngine:
    """Main bias detection engine"""
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        self.rule_based_detector = get_rule_based_detector() if self.config.enable_rule_based else None
        self.ml_classifier = get_ml_classifier() if self.config.enable_ml_classification else None
        self.intersectional_analyzer = IntersectionalAnalyzer() if self.config.enable_intersectional_analysis else None
        self.confidence_calculator = get_confidence_calculator()
        self.severity_calculator = get_severity_calculator()
        self.aggregated_scoring = get_aggregated_scoring()
        self.nlp = get_nlp_pipeline()
        self.taxonomy = get_taxonomy_loader()
        
    def detect_bias_spans(self, text: str, language: str = "auto") -> List[BiasDetectionResult]:
        """Main function to detect bias spans in text"""
        if not text or not text.strip():
            logger.warning("Empty text provided for bias detection")
            return []
        
        try:
            # Detect language if auto
            if language == "auto" and self.config.language_auto_detect:
                detected_lang, lang_confidence = self.nlp.detect_language(text)
                if lang_confidence > 0.7:
                    language = detected_lang
                else:
                    language = "en"  # Default to English
            
            # Initialize detection result
            result = BiasDetectionResult(
                id=str(uuid.uuid4()),
                text=text,
                language=language,
                detected_spans=[],
                overall_severity=0.0,
                overall_confidence=0.0,
                processing_metadata={
                    'detection_methods': [],
                    'processing_time': 0.0,
                    'config': self.config.__dict__
                },
                timestamp=datetime.now()
            )
            
            start_time = datetime.now()
            
            # Collect all detection spans
            all_spans = []
            
            # Rule-based detection
            if self.rule_based_detector:
                try:
                    rule_spans = self.rule_based_detector.detect_bias_spans(text, language)
                    all_spans.extend(rule_spans)
                    result.processing_metadata['detection_methods'].append('rule_based')
                    logger.info(f"Rule-based detection found {len(rule_spans)} spans")
                except Exception as e:
                    logger.error(f"Rule-based detection failed: {e}")
            
            # ML-based enhancement
            if self.ml_classifier:
                try:
                    enhanced_spans = self._enhance_with_ml_classification(all_spans, text)
                    all_spans = enhanced_spans
                    result.processing_metadata['detection_methods'].append('ml_classification')
                except Exception as e:
                    logger.error(f"ML classification failed: {e}")
            
            # Filter by thresholds
            filtered_spans = self._filter_spans(all_spans)
            
            # Merge overlapping spans if configured
            if self.config.merge_overlapping_spans:
                filtered_spans = self._merge_overlapping_spans(filtered_spans)
            
            # Limit number of spans
            if len(filtered_spans) > self.config.max_spans_per_text:
                # Sort by severity × confidence and take top N
                filtered_spans.sort(key=lambda x: x.severity * x.confidence, reverse=True)
                filtered_spans = filtered_spans[:self.config.max_spans_per_text]
            
            # Perform intersectional analysis
            intersectional_analysis = None
            if self.intersectional_analyzer:
                try:
                    intersectional_analysis = self.intersectional_analyzer.analyze_intersectional_bias(
                        text, filtered_spans
                    )
                    if intersectional_analysis:
                        result.processing_metadata['detection_methods'].append('intersectional')
                except Exception as e:
                    logger.error(f"Intersectional analysis failed: {e}")
            
            # Calculate overall scores
            overall_severity, overall_confidence = self._calculate_overall_scores(filtered_spans)
            
            # Finalize result
            result.detected_spans = filtered_spans
            result.overall_severity = overall_severity
            result.overall_confidence = overall_confidence
            result.intersectional_analysis = intersectional_analysis
            
            # Add processing metadata
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            result.processing_metadata['processing_time'] = processing_time
            result.processing_metadata['spans_detected'] = len(filtered_spans)
            result.processing_metadata['language_detected'] = language
            
            logger.info(f"Bias detection completed: {len(filtered_spans)} spans, "
                       f"severity {overall_severity:.2f}, confidence {overall_confidence:.2f}, "
                       f"time {processing_time:.2f}s")
            
            return [result]
            
        except Exception as e:
            logger.error(f"Bias detection failed: {e}")
            logger.error(traceback.format_exc())
            
            # Return error result
            return [BiasDetectionResult(
                id=str(uuid.uuid4()),
                text=text,
                language=language,
                detected_spans=[],
                overall_severity=0.0,
                overall_confidence=0.0,
                processing_metadata={'error': str(e)},
                timestamp=datetime.now()
            )]
    
    def _enhance_with_ml_classification(self, spans: List[BiasSpan], text: str) -> List[BiasSpan]:
        """Enhance spans with ML classification"""
        enhanced_spans = []
        
        for span in spans:
            try:
                # Get ML classification
                classification = self.ml_classifier.classify_span(
                    text, span.text, span.context_window
                )
                
                # Update span with ML confidence
                ml_confidence = self.ml_classifier.calculate_confidence(
                    span.text, span.bias_family, span.bias_subtype
                )
                
                # Combine confidences
                combined_confidence = (span.confidence + ml_confidence) / 2.0
                
                # Create enhanced span
                enhanced_span = BiasSpan(
                    start=span.start,
                    end=span.end,
                    text=span.text,
                    bias_family=classification.bias_family,
                    bias_subtype=classification.bias_subtype,
                    severity=span.severity,
                    confidence=combined_confidence,
                    method=DetectionMethod.HYBRID,
                    explanation=f"Rule-based + ML: {span.explanation}",
                    context_window=span.context_window,
                    intersectional_factors=span.intersectional_factors
                )
                
                enhanced_spans.append(enhanced_span)
                
            except Exception as e:
                logger.warning(f"ML enhancement failed for span: {e}")
                # Keep original span
                enhanced_spans.append(span)
        
        return enhanced_spans
    
    def _filter_spans(self, spans: List[BiasSpan]) -> List[BiasSpan]:
        """Filter spans based on thresholds"""
        filtered = []
        
        for span in spans:
            # Check confidence threshold
            if span.confidence < self.config.confidence_threshold:
                if not self.config.include_low_confidence:
                    continue
            
            # Check severity threshold
            if span.severity < self.config.severity_threshold:
                continue
            
            filtered.append(span)
        
        return filtered
    
    def _merge_overlapping_spans(self, spans: List[BiasSpan]) -> List[BiasSpan]:
        """Merge overlapping bias spans"""
        if not spans:
            return []
        
        # Sort by start position
        sorted_spans = sorted(spans, key=lambda x: x.start)
        
        merged = [sorted_spans[0]]
        
        for current in sorted_spans[1:]:
            last_merged = merged[-1]
            
            # Check for overlap (with small gap tolerance)
            gap_tolerance = 5  # characters
            if current.start <= last_merged.end + gap_tolerance:
                try:
                    # Merge spans
                    merged_span = last_merged.merge_with(current)
                    merged[-1] = merged_span
                except ValueError:
                    # If merge fails, keep separate
                    merged.append(current)
            else:
                merged.append(current)
        
        return merged
    
    def _calculate_overall_scores(self, spans: List[BiasSpan]) -> Tuple[float, float]:
        """Calculate overall severity and confidence scores"""
        if not spans:
            return 0.0, 0.0
        
        # Overall severity: weighted average by confidence
        total_weighted_severity = sum(span.severity * span.confidence for span in spans)
        total_confidence_weight = sum(span.confidence for span in spans)
        
        overall_severity = total_weighted_severity / total_confidence_weight if total_confidence_weight > 0 else 0.0
        
        # Overall confidence: maximum confidence among spans
        overall_confidence = max(span.confidence for span in spans)
        
        return overall_severity, overall_confidence
    
    def classify_bias_type(self, span: str, context: str) -> BiasClassification:
        """Classify bias type for a given span"""
        try:
            if self.ml_classifier:
                return self.ml_classifier.classify_span(context, span)
            else:
                # Fallback to rule-based classification
                return BiasClassification(
                    bias_family="cognitive",
                    bias_subtype="confirmation",
                    confidence=0.5,
                    evidence=["Fallback classification"]
                )
        except Exception as e:
            logger.error(f"Bias type classification failed: {e}")
            return BiasClassification(
                bias_family="cognitive",
                bias_subtype="confirmation",
                confidence=0.1,
                evidence=[f"Error: {str(e)}"]
            )
    
    def calculate_severity(self, bias_type: str, span: str) -> float:
        """Calculate severity for a bias type and span"""
        try:
            family, subtype = bias_type.split('.', 1) if '.' in bias_type else ('cognitive', 'confirmation')
            
            # Create minimal context
            context = {
                'has_generalization': 'all' in span.lower() or 'every' in span.lower(),
                'has_qualification': 'some' in span.lower() or 'maybe' in span.lower(),
                'explicit_language': any(word in span.lower() for word in ['hate', 'stupid', 'inferior'])
            }
            
            return self.severity_calculator.calculate_severity(
                family, subtype, context, method=self.config.severity_method
            )
        except Exception as e:
            logger.error(f"Severity calculation failed: {e}")
            return 5.0  # Default moderate severity
    
    def calculate_confidence(self, detection_signals: List) -> float:
        """Calculate confidence from detection signals"""
        try:
            # Convert generic signals to DetectionSignal objects if needed
            if not detection_signals:
                return 0.0
            
            signals = []
            for signal in detection_signals:
                if isinstance(signal, DetectionSignal):
                    signals.append(signal)
                else:
                    # Create DetectionSignal from generic data
                    signals.append(DetectionSignal(
                        method=DetectionMethod.RULE_BASED,
                        confidence=getattr(signal, 'confidence', 0.5),
                        severity=getattr(signal, 'severity', 5.0),
                        evidence=getattr(signal, 'evidence', 'Detection signal')
                    ))
            
            return self.confidence_calculator.calculate_confidence(
                signals, method=self.config.confidence_method
            )
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.0


# Global engine instance
_detection_engine = None


def get_detection_engine(config: Optional[DetectionConfig] = None) -> BiasDetectionEngine:
    """Get or create bias detection engine instance"""
    global _detection_engine
    if _detection_engine is None or config is not None:
        _detection_engine = BiasDetectionEngine(config)
    return _detection_engine


# Main detection functions (public API)
def detect_bias_spans(text: str, language: str = "auto") -> List[BiasDetectionResult]:
    """Detect bias spans in text - main public API function"""
    engine = get_detection_engine()
    return engine.detect_bias_spans(text, language)


def classify_bias_type(span: str, context: str) -> BiasClassification:
    """Classify bias type for a span - main public API function"""
    engine = get_detection_engine()
    return engine.classify_bias_type(span, context)


def calculate_severity(bias_type: str, span: str) -> float:
    """Calculate severity for bias - main public API function"""
    engine = get_detection_engine()
    return engine.calculate_severity(bias_type, span)


def calculate_confidence(detection_signals: List) -> float:
    """Calculate confidence from signals - main public API function"""
    engine = get_detection_engine()
    return engine.calculate_confidence(detection_signals)
