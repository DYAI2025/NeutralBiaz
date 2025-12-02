#!/usr/bin/env python3
"""
NLP Pipeline for Bias Detection

Provides comprehensive natural language processing capabilities including:
- Language detection
- Text preprocessing
- Tokenization and span detection
- spaCy and transformer model integration
- Multi-language support
"""

import logging
import re
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
from functools import lru_cache
import warnings

try:
    import spacy
    from spacy.lang.en import English
    from spacy.lang.de import German
except ImportError:
    spacy = None
    English = None
    German = None
    warnings.warn("spaCy not available. Install with: pip install spacy")

try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        pipeline, BertTokenizer, BertForSequenceClassification
    )
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    pipeline = None
    BertTokenizer = None
    BertForSequenceClassification = None
    warnings.warn("Transformers not available. Install with: pip install transformers torch")

try:
    import fasttext
except ImportError:
    fasttext = None
    warnings.warn("FastText not available. Install with: pip install fasttext")

from models.bias_models import BiasSpan, DetectionMethod


logger = logging.getLogger(__name__)


class LanguageDetector:
    """Detects language of input text"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self) -> None:
        """Load language detection model"""
        if fasttext is None:
            logger.warning("FastText not available, using simple heuristics for language detection")
            return
        
        try:
            if self.model_path and self.model_path.exists():
                self.model = fasttext.load_model(str(self.model_path))
            else:
                # Try to download model if not present
                import tempfile
                import os
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    model_file = os.path.join(temp_dir, "lid.176.bin")
                    if not os.path.exists(model_file):
                        logger.info("Downloading language identification model...")
                        fasttext.util.download_model('en', if_exists='ignore')
                    
                    if os.path.exists(model_file):
                        self.model = fasttext.load_model(model_file)
        except Exception as e:
            logger.warning(f"Could not load FastText model: {e}")
    
    def detect(self, text: str, confidence_threshold: float = 0.7) -> Tuple[str, float]:
        """Detect language of text"""
        if not text or not text.strip():
            return "en", 0.5
        
        if self.model:
            try:
                # Clean text for language detection
                clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
                predictions = self.model.predict(clean_text, k=1)
                language = predictions[0][0].replace('__label__', '')
                confidence = float(predictions[1][0])
                
                if confidence >= confidence_threshold:
                    return language, confidence
                else:
                    return "en", 0.5  # Default to English if uncertain
                    
            except Exception as e:
                logger.warning(f"FastText language detection failed: {e}")
        
        # Fallback to simple heuristics
        return self._detect_heuristic(text)
    
    def _detect_heuristic(self, text: str) -> Tuple[str, float]:
        """Simple heuristic-based language detection"""
        text_lower = text.lower()
        
        # German indicators
        german_indicators = ['der', 'die', 'das', 'und', 'ist', 'auf', 'mit', 'für', 'zu', 'ich', 'ein', 'eine']
        german_count = sum(1 for word in german_indicators if word in text_lower)
        
        # English indicators
        english_indicators = ['the', 'and', 'is', 'on', 'with', 'for', 'to', 'i', 'a', 'an']
        english_count = sum(1 for word in english_indicators if word in text_lower)
        
        if german_count > english_count and german_count > 0:
            return "de", min(0.8, german_count * 0.1)
        else:
            return "en", min(0.8, max(english_count * 0.1, 0.6))


class TextPreprocessor:
    """Preprocesses text for bias detection"""
    
    def __init__(self, language: str = "en"):
        self.language = language
    
    def clean_text(self, text: str) -> str:
        """Clean text while preserving structure for span detection"""
        if not text:
            return ""
        
        # Normalize whitespace but preserve structure
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for pattern matching"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Normalize punctuation
        text = re.sub(r'[\'‘’]', "'", text)  # Normalize quotes
        text = re.sub(r'[\u201c\u201d]', '"', text)  # Normalize double quotes
        text = re.sub(r'\u2013|\u2014', '-', text)  # Normalize dashes
        
        # Remove excessive punctuation
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        return text
    
    def extract_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Extract sentences with their positions"""
        if not text:
            return []
        
        # Simple sentence splitting with position tracking
        sentences = []
        current_pos = 0
        
        # Split on sentence endings
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        parts = re.split(sentence_pattern, text)
        
        for part in parts:
            part = part.strip()
            if part:
                start = text.find(part, current_pos)
                end = start + len(part)
                sentences.append((part, start, end))
                current_pos = end
        
        return sentences
    
    def extract_phrases(self, text: str, min_length: int = 3) -> List[Tuple[str, int, int]]:
        """Extract meaningful phrases"""
        if not text:
            return []
        
        phrases = []
        
        # Extract noun phrases, verb phrases, etc.
        # Simple approach: split on punctuation and conjunctions
        phrase_pattern = r'[,.;:]|\s+(?:and|or|but|however|therefore|moreover)\s+'
        parts = re.split(phrase_pattern, text)
        
        current_pos = 0
        for part in parts:
            part = part.strip()
            if len(part) >= min_length:
                start = text.find(part, current_pos)
                if start != -1:
                    end = start + len(part)
                    phrases.append((part, start, end))
                    current_pos = end
        
        return phrases


class SpacyProcessor:
    """spaCy-based NLP processing"""
    
    def __init__(self, language: str = "en"):
        self.language = language
        self.nlp = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load appropriate spaCy model"""
        if spacy is None:
            logger.warning("spaCy not available")
            return
        
        try:
            if self.language == "de":
                model_name = "de_core_news_sm"
            else:
                model_name = "en_core_web_sm"
            
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
            
        except OSError:
            logger.warning(f"spaCy model not found for {self.language}, using basic tokenizer")
            try:
                if self.language == "de":
                    self.nlp = German()
                else:
                    self.nlp = English()
            except Exception as e:
                logger.error(f"Could not create basic spaCy processor: {e}")
    
    def process(self, text: str) -> Optional[Any]:
        """Process text with spaCy"""
        if not self.nlp or not text:
            return None
        
        try:
            return self.nlp(text)
        except Exception as e:
            logger.error(f"spaCy processing failed: {e}")
            return None
    
    def extract_entities(self, text: str) -> List[Tuple[str, str, int, int]]:
        """Extract named entities"""
        doc = self.process(text)
        if not doc:
            return []
        
        entities = []
        for ent in doc.ents:
            entities.append((ent.text, ent.label_, ent.start_char, ent.end_char))
        
        return entities
    
    def extract_pos_tags(self, text: str) -> List[Tuple[str, str, int, int]]:
        """Extract part-of-speech tags"""
        doc = self.process(text)
        if not doc:
            return []
        
        pos_tags = []
        for token in doc:
            pos_tags.append((token.text, token.pos_, token.idx, token.idx + len(token.text)))
        
        return pos_tags
    
    def extract_dependencies(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract dependency relations"""
        doc = self.process(text)
        if not doc:
            return []
        
        dependencies = []
        for token in doc:
            dependencies.append((token.text, token.dep_, token.head.text))
        
        return dependencies


class TransformerProcessor:
    """Transformer-based processing for bias detection"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.classifier = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load transformer model"""
        if torch is None or AutoTokenizer is None:
            logger.warning("Transformers not available")
            return
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Create classifier pipeline
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info(f"Loaded transformer model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Could not load transformer model: {e}")
    
    def classify_text(self, text: str) -> Optional[Dict[str, float]]:
        """Classify text using transformer model"""
        if not self.classifier or not text:
            return None
        
        try:
            results = self.classifier(text)
            return {result['label']: result['score'] for result in results}
        except Exception as e:
            logger.error(f"Transformer classification failed: {e}")
            return None
    
    def encode_text(self, text: str) -> Optional[torch.Tensor]:
        """Encode text to embeddings"""
        if not self.tokenizer or not self.model or not text:
            return None
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                return outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            return None
    
    def similarity_search(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        emb1 = self.encode_text(text1)
        emb2 = self.encode_text(text2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        try:
            similarity = torch.cosine_similarity(emb1, emb2)
            return float(similarity.item())
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0


class NLPPipeline:
    """Complete NLP pipeline for bias detection"""
    
    def __init__(self, language: str = "auto", transformer_model: str = "distilbert-base-uncased"):
        self.default_language = language
        self.language_detector = LanguageDetector()
        self.preprocessors = {}
        self.spacy_processors = {}
        self.transformer_processor = TransformerProcessor(transformer_model)
        
    def get_preprocessor(self, language: str) -> TextPreprocessor:
        """Get or create text preprocessor for language"""
        if language not in self.preprocessors:
            self.preprocessors[language] = TextPreprocessor(language)
        return self.preprocessors[language]
    
    def get_spacy_processor(self, language: str) -> SpacyProcessor:
        """Get or create spaCy processor for language"""
        if language not in self.spacy_processors:
            self.spacy_processors[language] = SpacyProcessor(language)
        return self.spacy_processors[language]
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language of input text"""
        if self.default_language != "auto":
            return self.default_language, 1.0
        
        return self.language_detector.detect(text)
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text through complete NLP pipeline"""
        if not text or not text.strip():
            return {
                'text': text,
                'language': 'en',
                'language_confidence': 0.5,
                'cleaned_text': '',
                'sentences': [],
                'phrases': [],
                'entities': [],
                'pos_tags': [],
                'dependencies': [],
                'embeddings': None
            }
        
        # Detect language
        language, lang_confidence = self.detect_language(text)
        
        # Get processors
        preprocessor = self.get_preprocessor(language)
        spacy_processor = self.get_spacy_processor(language)
        
        # Preprocess
        cleaned_text = preprocessor.clean_text(text)
        normalized_text = preprocessor.normalize_text(text)
        
        # Extract linguistic structures
        sentences = preprocessor.extract_sentences(cleaned_text)
        phrases = preprocessor.extract_phrases(cleaned_text)
        
        # spaCy processing
        entities = spacy_processor.extract_entities(cleaned_text)
        pos_tags = spacy_processor.extract_pos_tags(cleaned_text)
        dependencies = spacy_processor.extract_dependencies(cleaned_text)
        
        # Transformer processing
        embeddings = self.transformer_processor.encode_text(cleaned_text)
        
        return {
            'text': text,
            'language': language,
            'language_confidence': lang_confidence,
            'cleaned_text': cleaned_text,
            'normalized_text': normalized_text,
            'sentences': sentences,
            'phrases': phrases,
            'entities': entities,
            'pos_tags': pos_tags,
            'dependencies': dependencies,
            'embeddings': embeddings
        }
    
    def extract_candidate_spans(self, text: str, min_length: int = 3, max_length: int = 100) -> List[Tuple[str, int, int]]:
        """Extract candidate text spans for bias detection"""
        processed = self.process_text(text)
        spans = []
        
        # Add sentences
        for sentence, start, end in processed['sentences']:
            if min_length <= len(sentence) <= max_length:
                spans.append((sentence, start, end))
        
        # Add phrases
        for phrase, start, end in processed['phrases']:
            if min_length <= len(phrase) <= max_length:
                spans.append((phrase, start, end))
        
        # Add entity contexts (entity + surrounding words)
        for entity_text, entity_type, start, end in processed['entities']:
            if entity_type in ['PERSON', 'ORG', 'NORP', 'GPE']:  # Relevant entity types
                # Extract context around entity
                context_start = max(0, start - 50)
                context_end = min(len(text), end + 50)
                context = text[context_start:context_end]
                
                if min_length <= len(context) <= max_length:
                    spans.append((context, context_start, context_end))
        
        # Remove duplicates and overlaps
        spans = self._remove_duplicate_spans(spans)
        
        return spans
    
    def _remove_duplicate_spans(self, spans: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
        """Remove duplicate and highly overlapping spans"""
        if not spans:
            return []
        
        # Sort by start position
        spans = sorted(spans, key=lambda x: x[1])
        
        filtered_spans = []
        for span in spans:
            text, start, end = span
            
            # Check for significant overlap with existing spans
            overlap = False
            for existing_text, existing_start, existing_end in filtered_spans:
                overlap_length = min(end, existing_end) - max(start, existing_start)
                if overlap_length > 0:
                    overlap_ratio = overlap_length / min(end - start, existing_end - existing_start)
                    if overlap_ratio > 0.7:  # 70% overlap threshold
                        overlap = True
                        break
            
            if not overlap:
                filtered_spans.append(span)
        
        return filtered_spans


# Global pipeline instance
_nlp_pipeline = None


@lru_cache(maxsize=1)
def get_nlp_pipeline(language: str = "auto", transformer_model: str = "distilbert-base-uncased") -> NLPPipeline:
    """Get or create NLP pipeline instance"""
    global _nlp_pipeline
    if _nlp_pipeline is None:
        _nlp_pipeline = NLPPipeline(language, transformer_model)
    return _nlp_pipeline
