#!/usr/bin/env python3
"""
ML-Based Bias Classifier

Implements machine learning-based bias classification using:
- Transformer models (BERT, DistilBERT, etc.)
- Fine-tuned bias detection models
- Ensemble classification
- Confidence estimation
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from functools import lru_cache
import warnings

try:
    import torch
    import torch.nn.functional as F
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        BertTokenizer, BertForSequenceClassification,
        DistilBertTokenizer, DistilBertForSequenceClassification,
        RobertaTokenizer, RobertaForSequenceClassification,
        pipeline
    )
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    warnings.warn("PyTorch and Transformers not available")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import cross_val_score
except ImportError:
    TfidfVectorizer = None
    LogisticRegression = None
    RandomForestClassifier = None
    warnings.warn("Scikit-learn not available")

from models.bias_models import BiasSpan, BiasClassification, DetectionMethod
from .taxonomy_loader import get_taxonomy_loader
from .nlp_pipeline import get_nlp_pipeline


logger = logging.getLogger(__name__)


class TransformerBiasClassifier:
    """Transformer-based bias classification"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", 
                 device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.classifier = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load transformer model for bias classification"""
        if torch is None or AutoTokenizer is None:
            logger.warning("PyTorch/Transformers not available")
            return
        
        try:
            # Load pre-trained model (we'll use a general classification model)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Move to device
            if torch.cuda.is_available() and self.device == "cuda":
                self.model = self.model.to(self.device)
            
            logger.info(f"Loaded transformer model: {self.model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")
    
    def encode_text(self, text: str, max_length: int = 512) -> Optional[torch.Tensor]:
        """Encode text to transformer embeddings"""
        if not self.tokenizer or not self.model or not text:
            return None
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            )
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use mean of last hidden state
                embeddings = outputs.hidden_states[-1].mean(dim=1)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return None
    
    def classify_bias_type(self, text: str, candidate_types: List[str]) -> List[Tuple[str, float]]:
        """Classify bias type using semantic similarity"""
        if not text or not candidate_types:
            return []
        
        try:
            # Get text embedding
            text_embedding = self.encode_text(text)
            if text_embedding is None:
                return []
            
            # Get embeddings for bias type descriptions
            taxonomy = get_taxonomy_loader()
            similarities = []
            
            for bias_type in candidate_types:
                # Parse family.subtype format
                if '.' in bias_type:
                    family_id, subtype_id = bias_type.split('.', 1)
                    family = taxonomy.get_family(family_id)
                    if family:
                        subtype = family.get_subtype(subtype_id)
                        if subtype:
                            # Create description for embedding
                            description = f"{family.name}: {subtype.name}. {subtype.description}"
                            type_embedding = self.encode_text(description)
                            
                            if type_embedding is not None:
                                # Calculate similarity
                                similarity = F.cosine_similarity(
                                    text_embedding, type_embedding, dim=1
                                ).item()
                                similarities.append((bias_type, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities
            
        except Exception as e:
            logger.error(f"Error classifying bias type: {e}")
            return []
    
    def calculate_bias_confidence(self, text: str, bias_type: str) -> float:
        """Calculate confidence for bias classification"""
        try:
            classifications = self.classify_bias_type(text, [bias_type])
            if classifications:
                return classifications[0][1]
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0


class HateSpeechClassifier:
    """Specialized classifier for hate speech detection"""
    
    def __init__(self):
        self.classifier = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load hate speech detection model"""
        if pipeline is None:
            logger.warning("Transformers pipeline not available")
            return
        
        try:
            # Use a pre-trained hate speech detection model
            self.classifier = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Loaded hate speech classifier")
        except Exception as e:
            logger.warning(f"Could not load hate speech classifier: {e}")
            # Fallback to general sentiment model
            try:
                self.classifier = pipeline(
                    "sentiment-analysis",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("Using sentiment analysis as fallback")
            except Exception as e2:
                logger.error(f"Could not load any classification model: {e2}")
    
    def classify_toxicity(self, text: str) -> Dict[str, float]:
        """Classify text toxicity"""
        if not self.classifier or not text:
            return {'toxicity': 0.0, 'confidence': 0.0}
        
        try:
            results = self.classifier(text)
            
            # Handle different model outputs
            if isinstance(results, list):
                result = results[0]
            else:
                result = results
            
            label = result.get('label', '').lower()
            score = result.get('score', 0.0)
            
            # Map labels to toxicity scores
            if label in ['toxic', 'negative', 'hate']:
                toxicity = score
            elif label in ['non-toxic', 'positive', 'non-hate']:
                toxicity = 1.0 - score
            else:
                toxicity = 0.5  # Uncertain
            
            return {
                'toxicity': toxicity,
                'confidence': score,
                'label': label
            }
            
        except Exception as e:
            logger.error(f"Error in toxicity classification: {e}")
            return {'toxicity': 0.0, 'confidence': 0.0}


class TraditionalMLClassifier:
    """Traditional ML classifier using TF-IDF and scikit-learn"""
    
    def __init__(self):
        self.vectorizer = None
        self.classifiers = {}
        self._setup_models()
    
    def _setup_models(self) -> None:
        """Setup traditional ML models"""
        if TfidfVectorizer is None or LogisticRegression is None:
            logger.warning("Scikit-learn not available")
            return
        
        try:
            # Setup TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 3),
                stop_words='english',
                lowercase=True,
                min_df=2,
                max_df=0.95
            )
            
            # Setup classifiers for different bias types
            self.classifiers = {
                'binary': LogisticRegression(random_state=42),
                'multiclass': RandomForestClassifier(n_estimators=100, random_state=42)
            }
            
            logger.info("Initialized traditional ML classifiers")
            
        except Exception as e:
            logger.error(f"Error setting up traditional ML models: {e}")
    
    def extract_features(self, texts: List[str]) -> Optional[np.ndarray]:
        """Extract TF-IDF features from texts"""
        if not self.vectorizer or not texts:
            return None
        
        try:
            # Fit and transform if not already fitted
            if not hasattr(self.vectorizer, 'vocabulary_'):
                features = self.vectorizer.fit_transform(texts)
            else:
                features = self.vectorizer.transform(texts)
            
            return features.toarray()
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def classify_binary(self, text: str) -> float:
        """Binary classification for bias presence"""
        if not text or 'binary' not in self.classifiers:
            return 0.0
        
        try:
            features = self.extract_features([text])
            if features is None:
                return 0.0
            
            # Use probability for bias class
            probabilities = self.classifiers['binary'].predict_proba(features)
            return float(probabilities[0][1]) if len(probabilities[0]) > 1 else 0.0
            
        except Exception as e:
            logger.error(f"Error in binary classification: {e}")
            return 0.0


class EnsembleBiasClassifier:
    """Ensemble classifier combining multiple approaches"""
    
    def __init__(self, transformer_model: str = "distilbert-base-uncased"):
        self.transformer_classifier = TransformerBiasClassifier(transformer_model)
        self.hate_speech_classifier = HateSpeechClassifier()
        self.traditional_classifier = TraditionalMLClassifier()
        self.nlp = get_nlp_pipeline()
        
        # Ensemble weights
        self.weights = {
            'transformer': 0.5,
            'hate_speech': 0.3,
            'traditional': 0.2
        }
    
    def classify_span(self, text: str, span_text: str, context_window: str = "") -> BiasClassification:
        """Classify a bias span using ensemble methods"""
        # Get all possible bias types
        taxonomy = get_taxonomy_loader()
        candidate_types = []
        
        for family_id, family in taxonomy.families.items():
            for subtype_id in family.subtypes.keys():
                candidate_types.append(f"{family_id}.{subtype_id}")
        
        # Get classifications from each method
        transformer_results = self._classify_with_transformer(span_text, candidate_types)
        hate_speech_results = self._classify_with_hate_speech(span_text)
        traditional_results = self._classify_with_traditional(span_text)
        
        # Combine results using ensemble weights
        final_scores = self._combine_classifications(
            transformer_results, hate_speech_results, traditional_results
        )
        
        # Get best classification
        if final_scores:
            best_type, best_score = max(final_scores.items(), key=lambda x: x[1])
            family, subtype = best_type.split('.', 1)
            
            return BiasClassification(
                bias_family=family,
                bias_subtype=subtype,
                confidence=best_score,
                evidence=[f"Ensemble classification with {len(final_scores)} candidates"],
                alternative_classifications=[
                    (t.split('.')[0], t.split('.')[1], s) 
                    for t, s in sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[1:4]
                ]
            )
        
        # Fallback classification
        return BiasClassification(
            bias_family="cognitive",
            bias_subtype="confirmation",
            confidence=0.1,
            evidence=["Fallback classification"],
            alternative_classifications=[]
        )
    
    def _classify_with_transformer(self, text: str, candidate_types: List[str]) -> Dict[str, float]:
        """Get transformer-based classifications"""
        try:
            results = self.transformer_classifier.classify_bias_type(text, candidate_types)
            return {bias_type: score for bias_type, score in results}
        except Exception as e:
            logger.error(f"Transformer classification failed: {e}")
            return {}
    
    def _classify_with_hate_speech(self, text: str) -> Dict[str, float]:
        """Get hate speech classifications"""
        try:
            toxicity_result = self.hate_speech_classifier.classify_toxicity(text)
            toxicity_score = toxicity_result.get('toxicity', 0.0)
            
            # Map toxicity to bias types
            results = {}
            if toxicity_score > 0.5:
                # High toxicity suggests certain bias types
                results['demographic.racial'] = toxicity_score * 0.8
                results['demographic.gender'] = toxicity_score * 0.7
                results['physical.disability'] = toxicity_score * 0.6
            
            return results
        except Exception as e:
            logger.error(f"Hate speech classification failed: {e}")
            return {}
    
    def _classify_with_traditional(self, text: str) -> Dict[str, float]:
        """Get traditional ML classifications"""
        try:
            bias_probability = self.traditional_classifier.classify_binary(text)
            
            # Distribute probability across common bias types
            results = {}
            if bias_probability > 0.3:
                results['demographic.racial'] = bias_probability * 0.3
                results['demographic.gender'] = bias_probability * 0.3
                results['socioeconomic.class'] = bias_probability * 0.2
                results['cultural.nationality'] = bias_probability * 0.2
            
            return results
        except Exception as e:
            logger.error(f"Traditional classification failed: {e}")
            return {}
    
    def _combine_classifications(self, transformer: Dict[str, float], 
                               hate_speech: Dict[str, float], 
                               traditional: Dict[str, float]) -> Dict[str, float]:
        """Combine classification results using ensemble weights"""
        combined = {}
        
        # Get all bias types from all methods
        all_types = set(transformer.keys()) | set(hate_speech.keys()) | set(traditional.keys())
        
        for bias_type in all_types:
            score = 0.0
            
            # Weight each method's contribution
            if bias_type in transformer:
                score += transformer[bias_type] * self.weights['transformer']
            
            if bias_type in hate_speech:
                score += hate_speech[bias_type] * self.weights['hate_speech']
            
            if bias_type in traditional:
                score += traditional[bias_type] * self.weights['traditional']
            
            if score > 0:
                combined[bias_type] = min(score, 1.0)  # Clamp to [0,1]
        
        return combined
    
    def calculate_confidence(self, text: str, bias_family: str, bias_subtype: str) -> float:
        """Calculate overall confidence for a bias classification"""
        bias_type = f"{bias_family}.{bias_subtype}"
        
        # Get individual confidences
        transformer_conf = self.transformer_classifier.calculate_bias_confidence(text, bias_type)
        hate_speech_result = self.hate_speech_classifier.classify_toxicity(text)
        hate_speech_conf = hate_speech_result.get('confidence', 0.0)
        traditional_conf = self.traditional_classifier.classify_binary(text)
        
        # Weighted combination
        total_confidence = (
            transformer_conf * self.weights['transformer'] +
            hate_speech_conf * self.weights['hate_speech'] +
            traditional_conf * self.weights['traditional']
        )
        
        return min(total_confidence, 1.0)


# Global classifier instance
_ml_classifier = None


@lru_cache(maxsize=1)
def get_ml_classifier(transformer_model: str = "distilbert-base-uncased") -> EnsembleBiasClassifier:
    """Get or create ML bias classifier instance"""
    global _ml_classifier
    if _ml_classifier is None:
        _ml_classifier = EnsembleBiasClassifier(transformer_model)
    return _ml_classifier
