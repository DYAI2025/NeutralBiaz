#!/usr/bin/env python3
"""
Bias Taxonomy Loader

Loads and validates bias taxonomy configurations from JSON files.
Provides centralized access to bias family definitions and validation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from functools import lru_cache

from models.bias_models import BiasFamily, BiasSubtype, validate_bias_family_config


logger = logging.getLogger(__name__)


class BiaxTaxonomyLoader:
    """Loads and manages bias taxonomy configurations"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent.parent.parent / "config" / "bias_families.json"
        self._families: Dict[str, BiasFamily] = {}
        self._intersectional_combinations: List[List[str]] = []
        self._severity_thresholds: Dict[str, float] = {}
        self._confidence_thresholds: Dict[str, float] = {}
        self._loaded = False
        
    def load_taxonomy(self) -> None:
        """Load bias taxonomy from configuration file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self._validate_config(config)
            self._load_families(config.get('bias_families', {}))
            self._intersectional_combinations = config.get('intersectional_combinations', [])
            self._severity_thresholds = config.get('severity_thresholds', {})
            self._confidence_thresholds = config.get('confidence_thresholds', {})
            
            self._loaded = True
            logger.info(f"Loaded {len(self._families)} bias families from {self.config_path}")
            
        except FileNotFoundError:
            logger.error(f"Bias taxonomy config file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in bias taxonomy config: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading bias taxonomy: {e}")
            raise
    
    def _validate_config(self, config: Dict) -> None:
        """Validate overall configuration structure"""
        required_sections = ['bias_families']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        if not isinstance(config['bias_families'], dict):
            raise ValueError("bias_families must be a dictionary")
    
    def _load_families(self, families_config: Dict) -> None:
        """Load bias families from configuration"""
        for family_id, family_config in families_config.items():
            try:
                validate_bias_family_config(family_config)
                
                # Load subtypes
                subtypes = {}
                for subtype_id, subtype_config in family_config.get('subtypes', {}).items():
                    subtypes[subtype_id] = BiasSubtype(**subtype_config)
                
                # Create family
                family = BiasFamily(
                    id=family_config['id'],
                    name=family_config['name'],
                    description=family_config['description'],
                    weight=family_config['weight'],
                    subtypes=subtypes,
                    intersectional_weights=family_config.get('intersectional_weights', {})
                )
                
                self._families[family_id] = family
                
            except Exception as e:
                logger.error(f"Error loading bias family '{family_id}': {e}")
                raise
    
    @property
    def families(self) -> Dict[str, BiasFamily]:
        """Get all loaded bias families"""
        if not self._loaded:
            self.load_taxonomy()
        return self._families
    
    @property
    def intersectional_combinations(self) -> List[List[str]]:
        """Get intersectional bias combinations"""
        if not self._loaded:
            self.load_taxonomy()
        return self._intersectional_combinations
    
    @property
    def severity_thresholds(self) -> Dict[str, float]:
        """Get severity level thresholds"""
        if not self._loaded:
            self.load_taxonomy()
        return self._severity_thresholds
    
    @property
    def confidence_thresholds(self) -> Dict[str, float]:
        """Get confidence level thresholds"""
        if not self._loaded:
            self.load_taxonomy()
        return self._confidence_thresholds
    
    def get_family(self, family_id: str) -> Optional[BiasFamily]:
        """Get a specific bias family by ID"""
        return self.families.get(family_id)
    
    def get_subtype(self, family_id: str, subtype_id: str) -> Optional[BiasSubtype]:
        """Get a specific bias subtype"""
        family = self.get_family(family_id)
        if family:
            return family.get_subtype(subtype_id)
        return None
    
    def get_all_patterns(self) -> Dict[str, List[str]]:
        """Get all bias patterns organized by family.subtype"""
        patterns = {}
        for family_id, family in self.families.items():
            for subtype_id, subtype in family.subtypes.items():
                key = f"{family_id}.{subtype_id}"
                patterns[key] = subtype.patterns
        return patterns
    
    def get_all_keywords(self) -> Dict[str, List[str]]:
        """Get all bias keywords organized by family.subtype"""
        keywords = {}
        for family_id, family in self.families.items():
            for subtype_id, subtype in family.subtypes.items():
                key = f"{family_id}.{subtype_id}"
                keywords[key] = getattr(subtype, 'keywords', [])
        return keywords
    
    def validate_bias_type(self, family_id: str, subtype_id: str) -> bool:
        """Validate that a bias type exists"""
        return self.get_subtype(family_id, subtype_id) is not None
    
    def get_intersectional_pairs(self) -> Set[tuple]:
        """Get all valid intersectional bias pairs"""
        pairs = set()
        for combination in self.intersectional_combinations:
            if len(combination) >= 2:
                for i in range(len(combination)):
                    for j in range(i + 1, len(combination)):
                        pairs.add((combination[i], combination[j]))
        return pairs
    
    def calculate_intersectional_amplification(self, bias_types: List[str]) -> float:
        """Calculate amplification factor for intersectional bias"""
        if len(bias_types) < 2:
            return 1.0
        
        # Base amplification increases with number of intersecting identities
        base_amplification = 1.0 + (len(bias_types) - 1) * 0.3
        
        # Additional amplification for known problematic combinations
        intersectional_pairs = self.get_intersectional_pairs()
        amplification_bonus = 0.0
        
        for i in range(len(bias_types)):
            for j in range(i + 1, len(bias_types)):
                if (bias_types[i], bias_types[j]) in intersectional_pairs:
                    amplification_bonus += 0.2
        
        return min(base_amplification + amplification_bonus, 3.0)  # Cap at 3x amplification
    
    def get_family_weights(self) -> Dict[str, float]:
        """Get weights for all bias families"""
        return {family_id: family.weight for family_id, family in self.families.items()}
    
    def get_family_names(self) -> Dict[str, str]:
        """Get human-readable names for all bias families"""
        return {family_id: family.name for family_id, family in self.families.items()}
    
    def get_subtype_names(self) -> Dict[str, Dict[str, str]]:
        """Get human-readable names for all bias subtypes"""
        names = {}
        for family_id, family in self.families.items():
            names[family_id] = {subtype_id: subtype.name for subtype_id, subtype in family.subtypes.items()}
        return names
    
    def search_patterns(self, query: str) -> List[Dict[str, str]]:
        """Search for patterns containing the query string"""
        results = []
        query_lower = query.lower()
        
        for family_id, family in self.families.items():
            for subtype_id, subtype in family.subtypes.items():
                for pattern in subtype.patterns:
                    if query_lower in pattern.lower():
                        results.append({
                            'family': family_id,
                            'subtype': subtype_id,
                            'pattern': pattern,
                            'family_name': family.name,
                            'subtype_name': subtype.name
                        })
        
        return results
    
    def get_statistics(self) -> Dict[str, int]:
        """Get taxonomy statistics"""
        total_subtypes = sum(len(family.subtypes) for family in self.families.values())
        total_patterns = sum(
            len(subtype.patterns) 
            for family in self.families.values() 
            for subtype in family.subtypes.values()
        )
        
        return {
            'families': len(self.families),
            'subtypes': total_subtypes,
            'patterns': total_patterns,
            'intersectional_combinations': len(self.intersectional_combinations)
        }


# Global instance
_taxonomy_loader = None


@lru_cache(maxsize=1)
def get_taxonomy_loader(config_path: Optional[Path] = None) -> BiaxTaxonomyLoader:
    """Get or create taxonomy loader instance"""
    global _taxonomy_loader
    if _taxonomy_loader is None:
        _taxonomy_loader = BiaxTaxonomyLoader(config_path)
    return _taxonomy_loader


def reload_taxonomy(config_path: Optional[Path] = None) -> BiaxTaxonomyLoader:
    """Force reload of taxonomy configuration"""
    global _taxonomy_loader
    _taxonomy_loader = BiaxTaxonomyLoader(config_path)
    _taxonomy_loader.load_taxonomy()
    get_taxonomy_loader.cache_clear()
    return _taxonomy_loader
