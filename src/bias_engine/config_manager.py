#!/usr/bin/env python3
"""
Configuration and Model Management System

Handles:
- Model loading and caching
- Configuration management
- Performance optimization
- Memory management
- Resource cleanup
"""

import logging
import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
import threading
import time
from functools import lru_cache
import psutil
import gc


logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for ML models"""
    name: str
    path: Optional[str] = None
    cache_size: int = 1
    load_on_startup: bool = False
    device: str = "auto"
    precision: str = "fp32"
    max_memory_mb: int = 1024
    timeout_seconds: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    max_parallel_requests: int = 10
    request_timeout: float = 30.0
    cache_size_mb: int = 512
    enable_gpu: bool = True
    batch_processing: bool = True
    max_batch_size: int = 32
    memory_cleanup_interval: int = 300  # seconds
    enable_profiling: bool = False


@dataclass
class BiasEngineConfig:
    """Main configuration for bias detection engine"""
    # Detection settings
    confidence_threshold: float = 0.3
    severity_threshold: float = 2.0
    max_text_length: int = 10000
    max_spans_per_text: int = 50
    enable_intersectional_analysis: bool = True
    enable_cultural_adaptation: bool = False
    
    # Model configurations
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    
    # Performance settings
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_metrics: bool = True
    
    # Language settings
    default_language: str = "en"
    supported_languages: List[str] = field(default_factory=lambda: ["en", "de"])
    
    # Data paths
    taxonomy_path: Optional[str] = None
    patterns_path: Optional[str] = None
    models_path: Optional[str] = None


class ModelCache:
    """Thread-safe model cache with memory management"""
    
    def __init__(self, max_size_mb: int = 512):
        self.max_size_mb = max_size_mb
        self._cache = {}
        self._access_times = {}
        self._memory_usage = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Any:
        """Get model from cache"""
        with self._lock:
            if key in self._cache:
                self._access_times[key] = time.time()
                return self._cache[key]
            return None
    
    def put(self, key: str, model: Any, size_mb: float = 0) -> None:
        """Put model in cache"""
        with self._lock:
            # Check memory limits
            if size_mb == 0:
                size_mb = self._estimate_model_size(model)
            
            # Clean cache if necessary
            self._cleanup_if_needed(size_mb)
            
            # Store model
            self._cache[key] = model
            self._access_times[key] = time.time()
            self._memory_usage[key] = size_mb
    
    def remove(self, key: str) -> bool:
        """Remove model from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._access_times[key]
                if key in self._memory_usage:
                    del self._memory_usage[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cached models"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._memory_usage.clear()
            gc.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_memory = sum(self._memory_usage.values())
            return {
                'cached_models': len(self._cache),
                'total_memory_mb': total_memory,
                'max_memory_mb': self.max_size_mb,
                'memory_usage_pct': (total_memory / self.max_size_mb) * 100,
                'models': list(self._cache.keys())
            }
    
    def _cleanup_if_needed(self, new_size_mb: float) -> None:
        """Clean up cache if memory limit would be exceeded"""
        current_usage = sum(self._memory_usage.values())
        
        if current_usage + new_size_mb > self.max_size_mb:
            # Remove least recently used models
            sorted_models = sorted(
                self._access_times.items(),
                key=lambda x: x[1]
            )
            
            for model_key, _ in sorted_models:
                if current_usage + new_size_mb <= self.max_size_mb:
                    break
                
                if model_key in self._memory_usage:
                    current_usage -= self._memory_usage[model_key]
                self.remove(model_key)
                logger.info(f"Removed cached model: {model_key}")
    
    def _estimate_model_size(self, model: Any) -> float:
        """Estimate model size in MB"""
        try:
            # Try to get model size if it's a torch model
            if hasattr(model, 'state_dict'):
                total_params = sum(p.numel() for p in model.parameters())
                # Estimate 4 bytes per parameter (float32)
                size_bytes = total_params * 4
                return size_bytes / (1024 * 1024)
            
            # Fallback: use sys.getsizeof
            import sys
            return sys.getsizeof(model) / (1024 * 1024)
        except:
            return 100.0  # Default estimate


class ConfigurationManager:
    """Manages configuration and models for bias detection engine"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.config = BiasEngineConfig()
        self.model_cache = ModelCache()
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        
        # Load configuration if provided
        if config_path:
            self.load_config(config_path)
        
        # Start memory cleanup thread
        self._start_cleanup_thread()
    
    def load_config(self, config_path: Path) -> None:
        """Load configuration from file"""
        try:
            if not config_path.exists():
                logger.warning(f"Config file not found: {config_path}")
                return
            
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Update configuration
            self._update_config_from_dict(config_data)
            
            # Update model cache size
            self.model_cache = ModelCache(self.config.performance.cache_size_mb)
            
            logger.info(f"Loaded configuration from: {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def save_config(self, config_path: Path) -> None:
        """Save configuration to file"""
        try:
            config_dict = asdict(self.config)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved configuration to: {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]) -> None:
        """Update configuration from dictionary"""
        # Update basic settings
        for key, value in config_data.items():
            if hasattr(self.config, key):
                if key == 'models':
                    # Handle model configurations
                    self.config.models = {
                        name: ModelConfig(**model_config)
                        for name, model_config in value.items()
                    }
                elif key == 'performance':
                    # Handle performance configuration
                    self.config.performance = PerformanceConfig(**value)
                else:
                    setattr(self.config, key, value)
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        return self.config.models.get(model_name)
    
    def add_model_config(self, model_name: str, config: ModelConfig) -> None:
        """Add model configuration"""
        self.config.models[model_name] = config
    
    def load_model(self, model_name: str, force_reload: bool = False) -> Any:
        """Load model with caching"""
        # Check cache first
        if not force_reload:
            cached_model = self.model_cache.get(model_name)
            if cached_model is not None:
                return cached_model
        
        # Get model configuration
        model_config = self.get_model_config(model_name)
        if not model_config:
            raise ValueError(f"Model configuration not found: {model_name}")
        
        try:
            # Load model based on type
            model = self._load_model_from_config(model_config)
            
            # Cache model
            if model_config.cache_size > 0:
                estimated_size = self._estimate_model_size(model)
                self.model_cache.put(model_name, model, estimated_size)
            
            logger.info(f"Loaded model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    def _load_model_from_config(self, config: ModelConfig) -> Any:
        """Load model based on configuration"""
        if 'transformers' in config.name.lower():
            return self._load_transformer_model(config)
        elif 'spacy' in config.name.lower():
            return self._load_spacy_model(config)
        elif 'fasttext' in config.name.lower():
            return self._load_fasttext_model(config)
        else:
            # Generic model loading
            return self._load_generic_model(config)
    
    def _load_transformer_model(self, config: ModelConfig) -> Any:
        """Load transformer model"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            model_path = config.path or config.name
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # Move to appropriate device
            device = self._get_device(config.device)
            if device != "cpu":
                model = model.to(device)
            
            return {'tokenizer': tokenizer, 'model': model, 'device': device}
            
        except ImportError:
            logger.error("Transformers library not available")
            raise
    
    def _load_spacy_model(self, config: ModelConfig) -> Any:
        """Load spaCy model"""
        try:
            import spacy
            
            model_name = config.path or config.name
            nlp = spacy.load(model_name)
            
            return nlp
            
        except ImportError:
            logger.error("spaCy library not available")
            raise
        except OSError:
            logger.error(f"spaCy model not found: {config.name}")
            raise
    
    def _load_fasttext_model(self, config: ModelConfig) -> Any:
        """Load FastText model"""
        try:
            import fasttext
            
            model_path = config.path
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"FastText model not found: {model_path}")
            
            model = fasttext.load_model(model_path)
            return model
            
        except ImportError:
            logger.error("FastText library not available")
            raise
    
    def _load_generic_model(self, config: ModelConfig) -> Any:
        """Load generic model"""
        # Placeholder for generic model loading
        logger.warning(f"Generic model loading not implemented for: {config.name}")
        return None
    
    def _get_device(self, device_config: str) -> str:
        """Determine appropriate device"""
        if device_config == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device_config
    
    def _estimate_model_size(self, model: Any) -> float:
        """Estimate model size in MB"""
        return self.model_cache._estimate_model_size(model)
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread"""
        def cleanup_worker():
            while not self._stop_cleanup.is_set():
                try:
                    self._periodic_cleanup()
                    time.sleep(self.config.performance.memory_cleanup_interval)
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _periodic_cleanup(self) -> None:
        """Perform periodic cleanup"""
        # Check memory usage
        memory_info = psutil.virtual_memory()
        if memory_info.percent > 85:  # If system memory > 85%
            logger.warning(f"High memory usage detected: {memory_info.percent}%")
            
            # Clear least recently used models
            cache_stats = self.model_cache.get_stats()
            if cache_stats['cached_models'] > 0:
                # Remove oldest model
                oldest_model = min(
                    self.model_cache._access_times.items(),
                    key=lambda x: x[1]
                )[0]
                self.model_cache.remove(oldest_model)
                logger.info(f"Removed cached model due to memory pressure: {oldest_model}")
        
        # Force garbage collection
        gc.collect()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system and configuration information"""
        memory_info = psutil.virtual_memory()
        cache_stats = self.model_cache.get_stats()
        
        return {
            'config': asdict(self.config),
            'system_memory': {
                'total_gb': round(memory_info.total / (1024**3), 2),
                'available_gb': round(memory_info.available / (1024**3), 2),
                'usage_pct': memory_info.percent
            },
            'model_cache': cache_stats,
            'supported_features': self._get_supported_features()
        }
    
    def _get_supported_features(self) -> Dict[str, bool]:
        """Check which optional features are available"""
        features = {}
        
        try:
            import torch
            features['pytorch'] = True
            features['cuda'] = torch.cuda.is_available()
        except ImportError:
            features['pytorch'] = False
            features['cuda'] = False
        
        try:
            import transformers
            features['transformers'] = True
        except ImportError:
            features['transformers'] = False
        
        try:
            import spacy
            features['spacy'] = True
        except ImportError:
            features['spacy'] = False
        
        try:
            import fasttext
            features['fasttext'] = True
        except ImportError:
            features['fasttext'] = False
        
        try:
            import sklearn
            features['sklearn'] = True
        except ImportError:
            features['sklearn'] = False
        
        return features
    
    def shutdown(self) -> None:
        """Cleanup resources"""
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
        
        self.model_cache.clear()
        logger.info("Configuration manager shut down")
    
    def __del__(self):
        self.shutdown()


# Global configuration manager
_config_manager = None


@lru_cache(maxsize=1)
def get_config_manager(config_path: Optional[Path] = None) -> ConfigurationManager:
    """Get or create configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager(config_path)
    return _config_manager


def create_default_config(save_path: Optional[Path] = None) -> BiasEngineConfig:
    """Create default configuration"""
    config = BiasEngineConfig(
        models={
            'distilbert-base': ModelConfig(
                name='distilbert-base-uncased',
                cache_size=1,
                load_on_startup=False,
                device='auto',
                max_memory_mb=512
            ),
            'spacy-en': ModelConfig(
                name='en_core_web_sm',
                cache_size=1,
                load_on_startup=True,
                max_memory_mb=256
            ),
            'spacy-de': ModelConfig(
                name='de_core_news_sm',
                cache_size=1,
                load_on_startup=False,
                max_memory_mb=256
            )
        }
    )
    
    if save_path:
        config_manager = ConfigurationManager()
        config_manager.config = config
        config_manager.save_config(save_path)
    
    return config
