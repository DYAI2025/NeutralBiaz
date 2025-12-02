#!/usr/bin/env python3
"""
Rule-Based Bias Detection Engine

Implements pattern-based bias detection using:
- Keyword matching
- Regular expression patterns
- Linguistic rule matching
- Context-aware detection
"""

import re
import logging
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass
from functools import lru_cache
import json

from models.bias_models import BiasSpan, BiasPatternMatch, DetectionMethod
from .taxonomy_loader import get_taxonomy_loader
from .nlp_pipeline import get_nlp_pipeline


logger = logging.getLogger(__name__)


@dataclass
class BiasPattern:
    """Represents a bias detection pattern"""
    pattern: str
    pattern_type: str  # 'keyword', 'regex', 'phrase', 'contextual'
    bias_family: str
    bias_subtype: str
    confidence_base: float
    context_required: bool = False
    case_sensitive: bool = False
    word_boundaries: bool = True
    severity_modifier: float = 1.0
    
    def __post_init__(self):
        if self.pattern_type == 'regex':
            try:
                re.compile(self.pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{self.pattern}': {e}")


class BiasPatternMatcher:
    """Matches bias patterns in text"""
    
    def __init__(self):
        self.patterns: List[BiasPattern] = []
        self.keyword_patterns: Dict[str, List[BiasPattern]] = {}
        self.regex_patterns: List[BiasPattern] = []
        self.phrase_patterns: List[BiasPattern] = []
        self.contextual_patterns: List[BiasPattern] = []
        self._load_patterns()
    
    def _load_patterns(self) -> None:
        """Load bias patterns from taxonomy and configuration"""
        taxonomy = get_taxonomy_loader()
        
        # Load patterns from bias families
        for family_id, family in taxonomy.families.items():
            for subtype_id, subtype in family.subtypes.items():
                # Basic keyword patterns
                for pattern in subtype.patterns:
                    bias_pattern = BiasPattern(
                        pattern=pattern,
                        pattern_type='keyword',
                        bias_family=family_id,
                        bias_subtype=subtype_id,
                        confidence_base=0.7,
                        severity_modifier=subtype.severity_multiplier
                    )
                    self.patterns.append(bias_pattern)
                    
                    # Index by first word for fast lookup
                    first_word = pattern.split()[0].lower()
                    if first_word not in self.keyword_patterns:
                        self.keyword_patterns[first_word] = []
                    self.keyword_patterns[first_word].append(bias_pattern)
        
        # Add specialized patterns
        self._add_specialized_patterns()
        
        logger.info(f"Loaded {len(self.patterns)} bias patterns")
    
    def _add_specialized_patterns(self) -> None:
        """Add specialized detection patterns"""
        # Gender bias patterns
        gender_patterns = [
            BiasPattern(
                pattern=r"\b(?:he|she)\s+(?:is|was)\s+(?:too\s+)?emotional\b",
                pattern_type='regex',
                bias_family='demographic',
                bias_subtype='gender',
                confidence_base=0.8,
                case_sensitive=False
            ),
            BiasPattern(
                pattern=r"\b(?:man|woman|boy|girl)\s+up\b",
                pattern_type='regex',
                bias_family='demographic',
                bias_subtype='gender',
                confidence_base=0.9,
                case_sensitive=False
            ),
            BiasPattern(
                pattern=r"\blike\s+a\s+(?:girl|woman)\b",
                pattern_type='regex',
                bias_family='demographic',
                bias_subtype='gender',
                confidence_base=0.85,
                case_sensitive=False
            )
        ]
        
        # Racial bias patterns
        racial_patterns = [
            BiasPattern(
                pattern=r"\b(?:all|most|typical)\s+(?:of\s+)?(?:them|those\s+people)\b",
                pattern_type='regex',
                bias_family='demographic',
                bias_subtype='racial',
                confidence_base=0.75,
                context_required=True
            ),
            BiasPattern(
                pattern=r"\bfor\s+(?:a|an)\s+\w+\s+(?:person|guy|girl)\b",
                pattern_type='regex',
                bias_family='demographic',
                bias_subtype='racial',
                confidence_base=0.7,
                context_required=True
            )
        ]
        
        # Age bias patterns
        age_patterns = [
            BiasPattern(
                pattern=r"\b(?:too\s+)?(?:old|young)\s+(?:to|for)\b",
                pattern_type='regex',
                bias_family='demographic',
                bias_subtype='age',
                confidence_base=0.8
            ),
            BiasPattern(
                pattern=r"\b(?:kids|youngsters)\s+these\s+days\b",
                pattern_type='regex',
                bias_family='demographic',
                bias_subtype='age',
                confidence_base=0.85
            )
        ]
        
        # Socioeconomic bias patterns
        socioeconomic_patterns = [
            BiasPattern(
                pattern=r"\bjust\s+(?:a|an)\s+\w+\s+(?:worker|employee|person)\b",
                pattern_type='regex',
                bias_family='socioeconomic',
                bias_subtype='occupation',
                confidence_base=0.75
            ),
            BiasPattern(
                pattern=r"\b(?:poor|rich)\s+people\s+(?:are|always|never)\b",
                pattern_type='regex',
                bias_family='socioeconomic',
                bias_subtype='class',
                confidence_base=0.9
            )
        ]
        
        # Cultural bias patterns
        cultural_patterns = [
            BiasPattern(
                pattern=r"\b(?:foreigners|immigrants|outsiders)\s+(?:are|always|never)\b",
                pattern_type='regex',
                bias_family='cultural',
                bias_subtype='nationality',
                confidence_base=0.9
            ),
            BiasPattern(
                pattern=r"\bthose\s+\w+\s+people\b",
                pattern_type='regex',
                bias_family='cultural',
                bias_subtype='nationality',
                confidence_base=0.7,
                context_required=True
            )
        ]
        
        # Disability bias patterns
        disability_patterns = [
            BiasPattern(
                pattern=r"\b(?:retarded|crippled|handicapped)\b",
                pattern_type='regex',
                bias_family='physical',
                bias_subtype='disability',
                confidence_base=0.95,
                severity_modifier=1.5
            ),
            BiasPattern(
                pattern=r"\bspecial\s+needs\b",
                pattern_type='regex',
                bias_family='physical',
                bias_subtype='disability',
                confidence_base=0.6,
                context_required=True
            )
        ]
        
        # Add all specialized patterns
        all_specialized = (
            gender_patterns + racial_patterns + age_patterns + 
            socioeconomic_patterns + cultural_patterns + disability_patterns
        )
        
        for pattern in all_specialized:
            self.patterns.append(pattern)
            if pattern.pattern_type == 'regex':
                self.regex_patterns.append(pattern)
            elif pattern.pattern_type == 'phrase':
                self.phrase_patterns.append(pattern)
            elif pattern.pattern_type == 'contextual':
                self.contextual_patterns.append(pattern)
    
    def find_keyword_matches(self, text: str, normalized_text: str) -> List[BiasPatternMatch]:
        """Find keyword-based pattern matches"""
        matches = []
        words = normalized_text.split()
        
        for i, word in enumerate(words):
            if word in self.keyword_patterns:
                for pattern in self.keyword_patterns[word]:
                    # Check if full pattern matches
                    pattern_words = pattern.pattern.split()
                    if i + len(pattern_words) <= len(words):
                        match_words = words[i:i + len(pattern_words)]
                        if match_words == pattern_words:
                            # Find position in original text
                            start = self._find_word_position(text, i, words)
                            end = self._find_word_position(text, i + len(pattern_words), words)
                            
                            if start != -1 and end != -1:
                                match = BiasPatternMatch(
                                    pattern=pattern.pattern,
                                    match_text=text[start:end],
                                    start=start,
                                    end=end,
                                    pattern_type='keyword',
                                    confidence=pattern.confidence_base
                                )
                                matches.append(match)
        
        return matches
    
    def find_regex_matches(self, text: str, normalized_text: str) -> List[BiasPatternMatch]:
        """Find regex-based pattern matches"""
        matches = []
        
        for pattern in self.regex_patterns:
            try:
                flags = 0 if pattern.case_sensitive else re.IGNORECASE
                if pattern.word_boundaries:
                    regex_pattern = rf"\b{pattern.pattern}\b"
                else:
                    regex_pattern = pattern.pattern
                
                for match in re.finditer(regex_pattern, normalized_text, flags):
                    # Map back to original text positions
                    start, end = self._map_to_original_positions(
                        text, normalized_text, match.start(), match.end()
                    )
                    
                    if start != -1 and end != -1:
                        pattern_match = BiasPatternMatch(
                            pattern=pattern.pattern,
                            match_text=text[start:end],
                            start=start,
                            end=end,
                            pattern_type='regex',
                            confidence=pattern.confidence_base
                        )
                        matches.append(pattern_match)
            
            except re.error as e:
                logger.warning(f"Regex error in pattern '{pattern.pattern}': {e}")
        
        return matches
    
    def _find_word_position(self, text: str, word_index: int, words: List[str]) -> int:
        """Find position of word in original text"""
        if word_index >= len(words):
            return len(text)
        
        current_pos = 0
        for i, word in enumerate(words):
            if i == word_index:
                # Find word starting from current position
                word_start = text.lower().find(word.lower(), current_pos)
                return word_start if word_start != -1 else current_pos
            else:
                # Move past this word
                word_pos = text.lower().find(word.lower(), current_pos)
                if word_pos != -1:
                    current_pos = word_pos + len(word)
        
        return -1
    
    def _map_to_original_positions(self, original: str, normalized: str, start: int, end: int) -> Tuple[int, int]:
        """Map positions from normalized text back to original text"""
        # Simple approach: find the matched text in original
        matched_text = normalized[start:end]
        
        # Try to find in original text (case-insensitive)
        original_lower = original.lower()
        normalized_lower = normalized.lower()
        matched_lower = matched_text.lower()
        
        # Find corresponding position in original
        pos = original_lower.find(matched_lower)
        if pos != -1:
            return pos, pos + len(matched_text)
        
        # Fallback: use approximate mapping
        ratio = len(original) / len(normalized) if len(normalized) > 0 else 1
        orig_start = int(start * ratio)
        orig_end = int(end * ratio)
        
        # Clamp to bounds
        orig_start = max(0, min(orig_start, len(original)))
        orig_end = max(orig_start, min(orig_end, len(original)))
        
        return orig_start, orig_end


class ContextualAnalyzer:
    """Analyzes context for bias detection"""
    
    def __init__(self):
        self.nlp = get_nlp_pipeline()
    
    def analyze_context(self, text: str, span_start: int, span_end: int, 
                       window_size: int = 100) -> Dict[str, Any]:
        """Analyze context around a potential bias span"""
        # Extract context window
        context_start = max(0, span_start - window_size)
        context_end = min(len(text), span_end + window_size)
        context_text = text[context_start:context_end]
        
        # Process context with NLP
        processed = self.nlp.process_text(context_text)
        
        # Analyze entities
        entities = processed.get('entities', [])
        person_entities = [ent for ent in entities if ent[1] in ['PERSON', 'ORG', 'NORP']]
        
        # Analyze sentiment/tone
        pos_tags = processed.get('pos_tags', [])
        adjectives = [tag for tag in pos_tags if tag[1] in ['ADJ', 'JJ', 'JJR', 'JJS']]
        
        # Check for intensifiers
        intensifiers = ['very', 'extremely', 'totally', 'completely', 'always', 'never', 'all']
        intensifier_count = sum(1 for word in context_text.lower().split() if word in intensifiers)
        
        # Check for qualifying language
        qualifiers = ['maybe', 'perhaps', 'possibly', 'sometimes', 'often', 'usually']
        qualifier_count = sum(1 for word in context_text.lower().split() if word in qualifiers)
        
        return {
            'context_text': context_text,
            'context_start': context_start,
            'context_end': context_end,
            'entities': entities,
            'person_entities': person_entities,
            'adjectives': adjectives,
            'intensifier_count': intensifier_count,
            'qualifier_count': qualifier_count,
            'has_generalization': intensifier_count > 0,
            'has_qualification': qualifier_count > 0
        }
    
    def calculate_context_confidence(self, context: Dict[str, Any], 
                                   base_confidence: float) -> float:
        """Calculate confidence modifier based on context"""
        modifier = 1.0
        
        # Increase confidence for generalizations
        if context.get('has_generalization', False):
            modifier += 0.2
        
        # Decrease confidence for qualifications
        if context.get('has_qualification', False):
            modifier -= 0.1
        
        # Increase confidence if targeting specific groups
        if context.get('person_entities', []):
            modifier += 0.1
        
        # Ensure confidence stays in valid range
        adjusted_confidence = base_confidence * modifier
        return max(0.0, min(1.0, adjusted_confidence))


class RuleBasedBiasDetector:
    """Complete rule-based bias detection system"""
    
    def __init__(self):
        self.pattern_matcher = BiasPatternMatcher()
        self.contextual_analyzer = ContextualAnalyzer()
        self.nlp = get_nlp_pipeline()
    
    def detect_bias_spans(self, text: str, language: str = "auto") -> List[BiasSpan]:
        """Detect bias spans using rule-based methods"""
        if not text or not text.strip():
            return []
        
        try:
            # Process text
            processed = self.nlp.process_text(text)
            normalized_text = processed.get('normalized_text', text.lower())
            
            # Find pattern matches
            keyword_matches = self.pattern_matcher.find_keyword_matches(text, normalized_text)
            regex_matches = self.pattern_matcher.find_regex_matches(text, normalized_text)
            
            all_matches = keyword_matches + regex_matches
            
            # Convert matches to bias spans
            bias_spans = []
            for match in all_matches:
                # Get pattern info
                pattern_info = self._get_pattern_info(match.pattern)
                if not pattern_info:
                    continue
                
                # Analyze context
                context = self.contextual_analyzer.analyze_context(
                    text, match.start, match.end
                )
                
                # Calculate final confidence
                final_confidence = self.contextual_analyzer.calculate_context_confidence(
                    context, match.confidence
                )
                
                # Skip low-confidence matches
                if final_confidence < 0.3:
                    continue
                
                # Calculate severity
                severity = self._calculate_severity(
                    pattern_info['family'], pattern_info['subtype'], 
                    context, pattern_info.get('severity_modifier', 1.0)
                )
                
                # Create bias span
                bias_span = BiasSpan(
                    start=match.start,
                    end=match.end,
                    text=match.match_text,
                    bias_family=pattern_info['family'],
                    bias_subtype=pattern_info['subtype'],
                    severity=severity,
                    confidence=final_confidence,
                    method=DetectionMethod.RULE_BASED,
                    explanation=f"Pattern match: {match.pattern}",
                    context_window=context.get('context_text', '')
                )
                
                bias_spans.append(bias_span)
            
            # Merge overlapping spans
            merged_spans = self._merge_overlapping_spans(bias_spans)
            
            logger.info(f"Detected {len(merged_spans)} bias spans using rule-based methods")
            return merged_spans
            
        except Exception as e:
            logger.error(f"Error in rule-based bias detection: {e}")
            return []
    
    def _get_pattern_info(self, pattern: str) -> Optional[Dict[str, Any]]:
        """Get pattern information from loaded patterns"""
        for bias_pattern in self.pattern_matcher.patterns:
            if bias_pattern.pattern == pattern:
                return {
                    'family': bias_pattern.bias_family,
                    'subtype': bias_pattern.bias_subtype,
                    'severity_modifier': bias_pattern.severity_modifier,
                    'confidence_base': bias_pattern.confidence_base
                }
        return None
    
    def _calculate_severity(self, family: str, subtype: str, context: Dict[str, Any], 
                          modifier: float) -> float:
        """Calculate severity score for bias span"""
        taxonomy = get_taxonomy_loader()
        
        # Get base severity from taxonomy
        base_severity = taxonomy.families[family].calculate_base_severity(subtype)
        
        # Apply context modifiers
        context_modifier = 1.0
        
        # Increase severity for generalizations
        if context.get('has_generalization', False):
            context_modifier += 0.3
        
        # Increase severity if targeting specific groups
        if context.get('person_entities', []):
            context_modifier += 0.2
        
        # Apply pattern-specific modifier
        final_severity = base_severity * modifier * context_modifier
        
        # Convert to 0-10 scale and clamp
        return max(0.0, min(10.0, final_severity * 5.0))  # Scale up to 0-10
    
    def _merge_overlapping_spans(self, spans: List[BiasSpan]) -> List[BiasSpan]:
        """Merge overlapping bias spans"""
        if not spans:
            return []
        
        # Sort by start position
        sorted_spans = sorted(spans, key=lambda x: x.start)
        
        merged = [sorted_spans[0]]
        
        for current in sorted_spans[1:]:
            last_merged = merged[-1]
            
            # Check for overlap
            if current.overlaps_with(last_merged):
                # Merge spans
                try:
                    merged_span = last_merged.merge_with(current)
                    merged[-1] = merged_span
                except ValueError:
                    # If merge fails, keep both spans
                    merged.append(current)
            else:
                merged.append(current)
        
        return merged


# Global detector instance
_rule_based_detector = None


@lru_cache(maxsize=1)
def get_rule_based_detector() -> RuleBasedBiasDetector:
    """Get or create rule-based bias detector instance"""
    global _rule_based_detector
    if _rule_based_detector is None:
        _rule_based_detector = RuleBasedBiasDetector()
    return _rule_based_detector
