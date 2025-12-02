#!/usr/bin/env python3
"""
Bias Detection Scoring Algorithms

Implements sophisticated scoring algorithms for:
- Confidence calculation based on multiple detection methods
- Severity assessment using intersectional and cultural factors
- Calibrated probability estimation
- Uncertainty quantification
"""

import logging
import math
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from models.bias_models import BiasSpan, DetectionMethod, IntersectionalAnalysis
from .taxonomy_loader import get_taxonomy_loader


logger = logging.getLogger(__name__)


class ConfidenceAggregationMethod(Enum):
    """Methods for aggregating confidence scores"""
    AVERAGE = "average"
    WEIGHTED_AVERAGE = "weighted_average"
    MAXIMUM = "maximum"
    ENSEMBLE = "ensemble"
    BAYESIAN = "bayesian"


class SeverityCalculationMethod(Enum):
    """Methods for calculating severity scores"""
    TAXONOMY_BASED = "taxonomy_based"
    FREQUENCY_WEIGHTED = "frequency_weighted"
    CONTEXT_AWARE = "context_aware"
    INTERSECTIONAL = "intersectional"
    CULTURAL_ADAPTIVE = "cultural_adaptive"


@dataclass
class DetectionSignal:
    """Represents a single bias detection signal"""
    method: DetectionMethod
    confidence: float
    severity: float
    evidence: str
    weight: float = 1.0
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if not 0.0 <= self.severity <= 10.0:
            raise ValueError(f"Severity must be between 0.0 and 10.0, got {self.severity}")
        if not 0.0 <= self.weight <= 5.0:
            raise ValueError(f"Weight must be between 0.0 and 5.0, got {self.weight}")


class ConfidenceCalculator:
    """Calculates confidence scores for bias detection"""
    
    def __init__(self):
        # Method weights for ensemble confidence
        self.method_weights = {
            DetectionMethod.RULE_BASED: 0.8,
            DetectionMethod.ML_CLASSIFICATION: 0.9,
            DetectionMethod.TRANSFORMER_MODEL: 1.0,
            DetectionMethod.HYBRID: 1.1,
            DetectionMethod.INTERSECTIONAL: 1.2
        }
        
        # Confidence calibration parameters
        self.calibration_params = {
            'alpha': 0.1,  # Platt scaling parameter
            'beta': 0.0,   # Platt scaling parameter
            'min_confidence': 0.05,  # Minimum reportable confidence
            'max_confidence': 0.95   # Maximum reportable confidence
        }
    
    def calculate_confidence(self, detection_signals: List[DetectionSignal], 
                           method: ConfidenceAggregationMethod = ConfidenceAggregationMethod.ENSEMBLE) -> float:
        """Calculate overall confidence from multiple detection signals"""
        if not detection_signals:
            return 0.0
        
        if method == ConfidenceAggregationMethod.AVERAGE:
            return self._calculate_average_confidence(detection_signals)
        elif method == ConfidenceAggregationMethod.WEIGHTED_AVERAGE:
            return self._calculate_weighted_average_confidence(detection_signals)
        elif method == ConfidenceAggregationMethod.MAXIMUM:
            return self._calculate_maximum_confidence(detection_signals)
        elif method == ConfidenceAggregationMethod.BAYESIAN:
            return self._calculate_bayesian_confidence(detection_signals)
        else:  # ENSEMBLE
            return self._calculate_ensemble_confidence(detection_signals)
    
    def _calculate_average_confidence(self, signals: List[DetectionSignal]) -> float:
        """Simple average of all confidence scores"""
        if not signals:
            return 0.0
        
        total_confidence = sum(signal.confidence for signal in signals)
        return total_confidence / len(signals)
    
    def _calculate_weighted_average_confidence(self, signals: List[DetectionSignal]) -> float:
        """Weighted average using signal weights"""
        if not signals:
            return 0.0
        
        weighted_sum = sum(signal.confidence * signal.weight for signal in signals)
        total_weight = sum(signal.weight for signal in signals)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_maximum_confidence(self, signals: List[DetectionSignal]) -> float:
        """Maximum confidence among all signals"""
        if not signals:
            return 0.0
        
        return max(signal.confidence for signal in signals)
    
    def _calculate_ensemble_confidence(self, signals: List[DetectionSignal]) -> float:
        """Ensemble confidence using method-specific weights"""
        if not signals:
            return 0.0
        
        # Group signals by method
        method_scores = {}
        for signal in signals:
            method = signal.method
            if method not in method_scores:
                method_scores[method] = []
            method_scores[method].append(signal.confidence)
        
        # Calculate weighted ensemble
        weighted_sum = 0.0
        total_weight = 0.0
        
        for method, confidences in method_scores.items():
            # Use maximum confidence for each method
            max_confidence = max(confidences)
            method_weight = self.method_weights.get(method, 1.0)
            
            weighted_sum += max_confidence * method_weight
            total_weight += method_weight
        
        ensemble_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Apply calibration
        return self._calibrate_confidence(ensemble_score)
    
    def _calculate_bayesian_confidence(self, signals: List[DetectionSignal]) -> float:
        """Bayesian confidence combination"""
        if not signals:
            return 0.0
        
        # Start with uniform prior
        log_odds = 0.0  # log(0.5 / (1 - 0.5)) = 0
        
        for signal in signals:
            # Convert confidence to likelihood ratio
            p = max(0.01, min(0.99, signal.confidence))  # Avoid extreme values
            likelihood_ratio = p / (1 - p)
            
            # Weight by method reliability
            weight = self.method_weights.get(signal.method, 1.0)
            
            # Add to log odds
            log_odds += weight * math.log(likelihood_ratio)
        
        # Convert back to probability
        odds = math.exp(log_odds)
        probability = odds / (1 + odds)
        
        return self._calibrate_confidence(probability)
    
    def _calibrate_confidence(self, raw_confidence: float) -> float:
        """Apply calibration to raw confidence score"""
        # Platt scaling calibration
        alpha = self.calibration_params['alpha']
        beta = self.calibration_params['beta']
        
        # Sigmoid calibration
        calibrated = 1.0 / (1.0 + math.exp(alpha * raw_confidence + beta))
        
        # Clamp to reasonable bounds
        min_conf = self.calibration_params['min_confidence']
        max_conf = self.calibration_params['max_confidence']
        
        return max(min_conf, min(max_conf, calibrated))
    
    def calculate_uncertainty(self, detection_signals: List[DetectionSignal]) -> Dict[str, float]:
        """Calculate various uncertainty measures"""
        if not detection_signals:
            return {'epistemic': 1.0, 'aleatoric': 1.0, 'total': 1.0}
        
        confidences = [signal.confidence for signal in signals]
        
        # Epistemic uncertainty (model uncertainty)
        epistemic = 1.0 - max(confidences) if confidences else 1.0
        
        # Aleatoric uncertainty (data uncertainty)
        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
        aleatoric = math.sqrt(variance)
        
        # Total uncertainty
        total = math.sqrt(epistemic ** 2 + aleatoric ** 2)
        
        return {
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'total': total
        }


class SeverityCalculator:
    """Calculates severity scores for bias detection"""
    
    def __init__(self):
        self.taxonomy = get_taxonomy_loader()
        
        # Severity scaling factors
        self.scaling_factors = {
            'context_amplification': 1.5,  # For contextual amplification
            'intersectional_boost': 2.0,   # For intersectional bias
            'explicit_penalty': 1.8,       # For explicit bias language
            'implicit_discount': 0.7,      # For implicit bias
            'frequency_weight': 1.3        # For frequent patterns
        }
        
        # Severity thresholds for different levels
        self.severity_levels = {
            'low': (0.0, 3.0),
            'medium': (3.0, 6.0),
            'high': (6.0, 8.0),
            'critical': (8.0, 10.0)
        }
    
    def calculate_severity(self, bias_family: str, bias_subtype: str, 
                         context: Dict[str, Any], 
                         intersectional_analysis: Optional[IntersectionalAnalysis] = None,
                         method: SeverityCalculationMethod = SeverityCalculationMethod.INTERSECTIONAL) -> float:
        """Calculate severity score for bias detection"""
        
        if method == SeverityCalculationMethod.TAXONOMY_BASED:
            return self._calculate_taxonomy_severity(bias_family, bias_subtype)
        elif method == SeverityCalculationMethod.FREQUENCY_WEIGHTED:
            return self._calculate_frequency_weighted_severity(bias_family, bias_subtype, context)
        elif method == SeverityCalculationMethod.CONTEXT_AWARE:
            return self._calculate_context_aware_severity(bias_family, bias_subtype, context)
        elif method == SeverityCalculationMethod.CULTURAL_ADAPTIVE:
            return self._calculate_cultural_adaptive_severity(bias_family, bias_subtype, context)
        else:  # INTERSECTIONAL
            return self._calculate_intersectional_severity(
                bias_family, bias_subtype, context, intersectional_analysis
            )
    
    def _calculate_taxonomy_severity(self, bias_family: str, bias_subtype: str) -> float:
        """Base severity from taxonomy definition"""
        family = self.taxonomy.get_family(bias_family)
        if not family:
            return 5.0  # Default moderate severity
        
        base_severity = family.calculate_base_severity(bias_subtype)
        
        # Scale to 0-10 range
        return min(10.0, base_severity * 5.0)
    
    def _calculate_frequency_weighted_severity(self, bias_family: str, bias_subtype: str, 
                                             context: Dict[str, Any]) -> float:
        """Severity weighted by pattern frequency"""
        base_severity = self._calculate_taxonomy_severity(bias_family, bias_subtype)
        
        # Check for repeated patterns (frequency boost)
        pattern_count = context.get('pattern_matches', 1)
        frequency_multiplier = min(2.0, 1.0 + (pattern_count - 1) * 0.2)
        
        return min(10.0, base_severity * frequency_multiplier)
    
    def _calculate_context_aware_severity(self, bias_family: str, bias_subtype: str, 
                                        context: Dict[str, Any]) -> float:
        """Severity adjusted based on context"""
        base_severity = self._calculate_taxonomy_severity(bias_family, bias_subtype)
        
        # Context modifiers
        context_multiplier = 1.0
        
        # Amplify for generalizations
        if context.get('has_generalization', False):
            context_multiplier *= self.scaling_factors['context_amplification']
        
        # Amplify for explicit language
        if context.get('explicit_language', False):
            context_multiplier *= self.scaling_factors['explicit_penalty']
        
        # Reduce for qualified statements
        if context.get('has_qualification', False):
            context_multiplier *= self.scaling_factors['implicit_discount']
        
        # Amplify if targeting specific groups
        if context.get('targets_specific_group', False):
            context_multiplier *= 1.3
        
        return min(10.0, base_severity * context_multiplier)
    
    def _calculate_intersectional_severity(self, bias_family: str, bias_subtype: str, 
                                         context: Dict[str, Any],
                                         intersectional_analysis: Optional[IntersectionalAnalysis]) -> float:
        """Severity with intersectional bias considerations"""
        base_severity = self._calculate_context_aware_severity(bias_family, bias_subtype, context)
        
        if not intersectional_analysis:
            return base_severity
        
        # Apply intersectional amplification
        intersectional_multiplier = intersectional_analysis.amplification_factor
        
        # Additional boost for multiple identity intersections
        num_identities = len(intersectional_analysis.detected_identities)
        if num_identities > 1:
            identity_boost = 1.0 + (num_identities - 1) * 0.2
            intersectional_multiplier *= identity_boost
        
        # Boost for marginalization indicators
        if intersectional_analysis.marginalization_indicators:
            marginalization_boost = 1.0 + len(intersectional_analysis.marginalization_indicators) * 0.15
            intersectional_multiplier *= marginalization_boost
        
        return min(10.0, base_severity * intersectional_multiplier)
    
    def _calculate_cultural_adaptive_severity(self, bias_family: str, bias_subtype: str, 
                                            context: Dict[str, Any]) -> float:
        """Severity adapted for cultural context"""
        base_severity = self._calculate_context_aware_severity(bias_family, bias_subtype, context)
        
        # Cultural modifiers from context
        cultural_modifiers = context.get('cultural_modifiers', {})
        
        cultural_multiplier = 1.0
        for modifier_type, modifier_value in cultural_modifiers.items():
            if modifier_type == 'power_distance':
                # Higher severity in low power distance cultures
                cultural_multiplier *= (2.0 - modifier_value)
            elif modifier_type == 'individualism':
                # Higher severity in individualistic cultures for certain bias types
                if bias_family in ['demographic', 'physical']:
                    cultural_multiplier *= (1.0 + modifier_value * 0.3)
            elif modifier_type == 'uncertainty_avoidance':
                # Higher severity in high uncertainty avoidance cultures
                cultural_multiplier *= (1.0 + modifier_value * 0.2)
        
        return min(10.0, base_severity * cultural_multiplier)
    
    def get_severity_level(self, severity_score: float) -> str:
        """Convert numeric severity to categorical level"""
        for level, (min_val, max_val) in self.severity_levels.items():
            if min_val <= severity_score < max_val:
                return level
        return 'critical'  # For scores >= 8.0
    
    def calculate_relative_severity(self, spans: List[BiasSpan]) -> List[float]:
        """Calculate relative severity scores within a set of spans"""
        if not spans:
            return []
        
        severities = [span.severity for span in spans]
        max_severity = max(severities)
        min_severity = min(severities)
        
        if max_severity == min_severity:
            return [1.0] * len(spans)  # All equal
        
        # Normalize to 0-1 range
        relative_severities = [
            (severity - min_severity) / (max_severity - min_severity)
            for severity in severities
        ]
        
        return relative_severities


class AggregatedScoring:
    """Combines confidence and severity scoring"""
    
    def __init__(self):
        self.confidence_calculator = ConfidenceCalculator()
        self.severity_calculator = SeverityCalculator()
    
    def calculate_detection_score(self, detection_signals: List[DetectionSignal], 
                                bias_family: str, bias_subtype: str,
                                context: Dict[str, Any],
                                intersectional_analysis: Optional[IntersectionalAnalysis] = None) -> Dict[str, float]:
        """Calculate comprehensive detection score"""
        
        # Calculate confidence
        confidence = self.confidence_calculator.calculate_confidence(detection_signals)
        uncertainty = self.confidence_calculator.calculate_uncertainty(detection_signals)
        
        # Calculate severity
        severity = self.severity_calculator.calculate_severity(
            bias_family, bias_subtype, context, intersectional_analysis
        )
        
        # Calculate combined risk score
        risk_score = self._calculate_risk_score(confidence, severity)
        
        # Calculate priority score for action ranking
        priority_score = self._calculate_priority_score(confidence, severity, context)
        
        return {
            'confidence': confidence,
            'severity': severity,
            'risk_score': risk_score,
            'priority_score': priority_score,
            'uncertainty_total': uncertainty['total'],
            'uncertainty_epistemic': uncertainty['epistemic'],
            'uncertainty_aleatoric': uncertainty['aleatoric']
        }
    
    def _calculate_risk_score(self, confidence: float, severity: float) -> float:
        """Calculate risk score combining confidence and severity"""
        # Risk = Probability × Impact
        # Normalize severity to 0-1 scale
        normalized_severity = severity / 10.0
        
        # Risk is product of confidence (probability) and normalized severity (impact)
        risk = confidence * normalized_severity
        
        # Apply scaling for extreme combinations
        if confidence > 0.8 and normalized_severity > 0.8:
            risk *= 1.2  # Boost high-confidence, high-severity cases
        elif confidence < 0.3 or normalized_severity < 0.3:
            risk *= 0.8  # Reduce low-confidence or low-severity cases
        
        return min(1.0, risk)
    
    def _calculate_priority_score(self, confidence: float, severity: float, 
                                context: Dict[str, Any]) -> float:
        """Calculate priority score for action ranking"""
        base_priority = self._calculate_risk_score(confidence, severity)
        
        # Priority modifiers based on context
        priority_multiplier = 1.0
        
        # Boost priority for public-facing content
        if context.get('is_public', False):
            priority_multiplier *= 1.3
        
        # Boost priority for targeting vulnerable groups
        if context.get('targets_vulnerable_group', False):
            priority_multiplier *= 1.4
        
        # Boost priority for widespread content
        reach = context.get('content_reach', 1)
        if reach > 100:
            priority_multiplier *= min(2.0, 1.0 + math.log10(reach / 100))
        
        # Reduce priority for historical content
        if context.get('is_historical', False):
            priority_multiplier *= 0.7
        
        return min(1.0, base_priority * priority_multiplier)


# Global scoring instances
_confidence_calculator = None
_severity_calculator = None
_aggregated_scoring = None


def get_confidence_calculator() -> ConfidenceCalculator:
    """Get or create confidence calculator instance"""
    global _confidence_calculator
    if _confidence_calculator is None:
        _confidence_calculator = ConfidenceCalculator()
    return _confidence_calculator


def get_severity_calculator() -> SeverityCalculator:
    """Get or create severity calculator instance"""
    global _severity_calculator
    if _severity_calculator is None:
        _severity_calculator = SeverityCalculator()
    return _severity_calculator


def get_aggregated_scoring() -> AggregatedScoring:
    """Get or create aggregated scoring instance"""
    global _aggregated_scoring
    if _aggregated_scoring is None:
        _aggregated_scoring = AggregatedScoring()
    return _aggregated_scoring
