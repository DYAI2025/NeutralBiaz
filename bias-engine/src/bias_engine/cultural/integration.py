"""
Cultural Integration Module

Provides seamless integration hooks for the cultural adaptation engine
with bias detection systems and other components.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from functools import wraps

from .models.hofstede_model import HofstedeModel
from .data.profile_manager import CulturalProfileManager
from .adapters.cultural_adapter import CulturalAdapter
from .analyzers.cultural_analyzer import CulturalAnalyzer
from .intelligence.cultural_intelligence import CulturalIntelligence


class CulturalIntegrationError(Exception):
    """Custom exception for cultural integration errors."""
    pass


class CulturalIntegration:
    """
    Main integration interface for the cultural adaptation engine.

    Provides high-level methods for integrating cultural analysis
    with bias detection systems and other components.
    """

    def __init__(self,
                 profile_manager: Optional[CulturalProfileManager] = None,
                 enable_caching: bool = True):
        """
        Initialize the cultural integration system.

        Args:
            profile_manager: Optional custom profile manager
            enable_caching: Whether to enable result caching
        """
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.profile_manager = profile_manager or CulturalProfileManager()
        self.hofstede_model = HofstedeModel()
        self.cultural_adapter = CulturalAdapter(self.profile_manager)
        self.cultural_analyzer = CulturalAnalyzer(self.profile_manager)
        self.cultural_intelligence = CulturalIntelligence(self.profile_manager)

        self.enable_caching = enable_caching
        self._cache = {} if enable_caching else None

        # Integration hooks
        self._pre_bias_hooks: List[Callable] = []
        self._post_bias_hooks: List[Callable] = []

    def enhance_bias_detection(self,
                               bias_results: Dict[str, Any],
                               sender_culture: str,
                               receiver_culture: str,
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main entry point for cultural enhancement of bias detection results.

        Args:
            bias_results: Original bias detection results
            sender_culture: Sender's culture code
            receiver_culture: Receiver's culture code
            context: Optional contextual information

        Returns:
            Culturally-enhanced bias detection results
        """
        try:
            cache_key = None
            if self.enable_caching:
                cache_key = f"enhance_{sender_culture}_{receiver_culture}_{hash(str(bias_results))}"
                if cache_key in self._cache:
                    self.logger.debug(f"Returning cached result for {cache_key}")
                    return self._cache[cache_key]

            # Execute pre-bias hooks
            for hook in self._pre_bias_hooks:
                try:
                    hook(bias_results, sender_culture, receiver_culture, context)
                except Exception as e:
                    self.logger.warning(f"Pre-bias hook failed: {e}")

            # Apply cultural modifiers to bias results
            enhanced_results = self.cultural_adapter.apply_cultural_modifiers(
                bias_results, sender_culture, receiver_culture
            )

            # Add cross-cultural analysis
            cultural_analysis = self.cultural_analyzer.analyze_cross_cultural_context(
                sender_culture, receiver_culture, context
            )

            enhanced_results["cross_cultural_analysis"] = {
                "overall_risk": cultural_analysis.overall_risk.value,
                "cultural_distance": cultural_analysis.cultural_distance,
                "bridge_score": cultural_analysis.bridge_score,
                "insights": [insight.__dict__ for insight in cultural_analysis.insights],
                "recommendations": cultural_analysis.recommendations,
                "potential_misunderstandings": cultural_analysis.potential_misunderstandings
            }

            # Add cultural intelligence insights
            sensitivity_warnings = self.cultural_intelligence.generate_sensitivity_warnings(
                sender_culture, receiver_culture
            )

            communication_recommendations = self.cultural_intelligence.recommend_communication_strategies(
                sender_culture, receiver_culture, context.get("type") if context else None
            )

            enhanced_results["cultural_intelligence"] = {
                "sensitivity_warnings": sensitivity_warnings,
                "communication_strategies": communication_recommendations
            }

            # Execute post-bias hooks
            for hook in self._post_bias_hooks:
                try:
                    hook(enhanced_results, sender_culture, receiver_culture, context)
                except Exception as e:
                    self.logger.warning(f"Post-bias hook failed: {e}")

            # Cache result if caching is enabled
            if self.enable_caching and cache_key:
                self._cache[cache_key] = enhanced_results

            return enhanced_results

        except Exception as e:
            self.logger.error(f"Error in cultural enhancement: {e}")
            raise CulturalIntegrationError(f"Cultural enhancement failed: {e}")

    def analyze_cultural_context(self,
                                 sender_culture: str,
                                 receiver_culture: str,
                                 context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze cultural context without bias results.

        Args:
            sender_culture: Sender's culture code
            receiver_culture: Receiver's culture code
            context: Optional contextual information

        Returns:
            Cultural context analysis results
        """
        try:
            cache_key = None
            if self.enable_caching:
                cache_key = f"context_{sender_culture}_{receiver_culture}_{hash(str(context))}"
                if cache_key in self._cache:
                    return self._cache[cache_key]

            analysis = self.cultural_analyzer.analyze_cross_cultural_context(
                sender_culture, receiver_culture, context
            )

            communication_strategies = self.cultural_intelligence.recommend_communication_strategies(
                sender_culture, receiver_culture, context.get("type") if context else None
            )

            sensitivity_warnings = self.cultural_intelligence.generate_sensitivity_warnings(
                sender_culture, receiver_culture
            )

            result = {
                "cultural_distance": analysis.cultural_distance,
                "bridge_score": analysis.bridge_score,
                "overall_risk": analysis.overall_risk.value,
                "insights": [insight.__dict__ for insight in analysis.insights],
                "recommendations": analysis.recommendations,
                "potential_misunderstandings": analysis.potential_misunderstandings,
                "communication_strategies": communication_strategies,
                "sensitivity_warnings": sensitivity_warnings
            }

            if self.enable_caching and cache_key:
                self._cache[cache_key] = result

            return result

        except Exception as e:
            self.logger.error(f"Error in cultural context analysis: {e}")
            raise CulturalIntegrationError(f"Cultural context analysis failed: {e}")

    def get_cultural_dashboard_data(self, culture_codes: List[str]) -> Dict[str, Any]:
        """
        Generate comprehensive cultural dashboard data.

        Args:
            culture_codes: List of culture codes to include

        Returns:
            Dashboard-ready cultural data
        """
        try:
            return self.cultural_intelligence.generate_cultural_dashboard_data(culture_codes)
        except Exception as e:
            self.logger.error(f"Error generating dashboard data: {e}")
            raise CulturalIntegrationError(f"Dashboard data generation failed: {e}")

    def register_pre_bias_hook(self, hook_func: Callable) -> None:
        """
        Register a hook to be called before bias enhancement.

        Args:
            hook_func: Function to call before bias enhancement
        """
        if callable(hook_func):
            self._pre_bias_hooks.append(hook_func)
        else:
            raise ValueError("Hook function must be callable")

    def register_post_bias_hook(self, hook_func: Callable) -> None:
        """
        Register a hook to be called after bias enhancement.

        Args:
            hook_func: Function to call after bias enhancement
        """
        if callable(hook_func):
            self._post_bias_hooks.append(hook_func)
        else:
            raise ValueError("Hook function must be callable")

    def clear_cache(self) -> None:
        """Clear the integration cache."""
        if self._cache:
            self._cache.clear()
            self.logger.info("Cultural integration cache cleared")

    def get_supported_cultures(self) -> List[str]:
        """Get list of supported culture codes."""
        return self.profile_manager.get_supported_cultures()

    def validate_culture_codes(self, *culture_codes: str) -> bool:
        """
        Validate that culture codes are supported.

        Args:
            culture_codes: Culture codes to validate

        Returns:
            True if all codes are valid, False otherwise
        """
        supported = set(self.get_supported_cultures())
        for code in culture_codes:
            if code.upper() not in supported:
                self.logger.warning(f"Culture code {code} not supported")
                return False
        return True

    def get_cultural_statistics(self) -> Dict[str, Any]:
        """Get statistics about the cultural system."""
        return {
            "supported_cultures": len(self.get_supported_cultures()),
            "profile_manager_stats": self.profile_manager.get_statistics(),
            "cache_size": len(self._cache) if self._cache else 0,
            "hooks": {
                "pre_bias": len(self._pre_bias_hooks),
                "post_bias": len(self._post_bias_hooks)
            }
        }


def with_cultural_context(sender_culture_key: str = "sender_culture",
                          receiver_culture_key: str = "receiver_culture",
                          context_key: str = "context"):
    """
    Decorator for automatically adding cultural context to function results.

    Args:
        sender_culture_key: Key name for sender culture in kwargs
        receiver_culture_key: Key name for receiver culture in kwargs
        context_key: Key name for context in kwargs

    Returns:
        Decorated function with cultural context
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get cultural parameters from kwargs
            sender_culture = kwargs.get(sender_culture_key)
            receiver_culture = kwargs.get(receiver_culture_key)
            context = kwargs.get(context_key)

            # Execute original function
            result = func(*args, **kwargs)

            # Add cultural context if culture parameters are provided
            if sender_culture and receiver_culture and isinstance(result, dict):
                cultural_integration = CulturalIntegration()
                cultural_context = cultural_integration.analyze_cultural_context(
                    sender_culture, receiver_culture, context
                )
                result["cultural_context"] = cultural_context

            return result
        return wrapper
    return decorator


def cultural_bias_enhancer(bias_detection_func: Callable) -> Callable:
    """
    Decorator for automatically enhancing bias detection with cultural context.

    Args:
        bias_detection_func: Original bias detection function

    Returns:
        Enhanced bias detection function
    """
    @wraps(bias_detection_func)
    def wrapper(*args, **kwargs):
        # Execute original bias detection
        bias_results = bias_detection_func(*args, **kwargs)

        # Check for cultural parameters in kwargs
        sender_culture = kwargs.get("sender_culture")
        receiver_culture = kwargs.get("receiver_culture")
        context = kwargs.get("context")

        # Apply cultural enhancement if parameters are available
        if sender_culture and receiver_culture and isinstance(bias_results, dict):
            cultural_integration = CulturalIntegration()
            enhanced_results = cultural_integration.enhance_bias_detection(
                bias_results, sender_culture, receiver_culture, context
            )
            return enhanced_results

        return bias_results
    return wrapper


# Global instance for easy access
_global_cultural_integration = None


def get_cultural_integration() -> CulturalIntegration:
    """Get the global cultural integration instance."""
    global _global_cultural_integration
    if _global_cultural_integration is None:
        _global_cultural_integration = CulturalIntegration()
    return _global_cultural_integration


def initialize_cultural_integration(**kwargs) -> CulturalIntegration:
    """
    Initialize the global cultural integration instance with custom parameters.

    Args:
        **kwargs: Initialization parameters for CulturalIntegration

    Returns:
        Initialized CulturalIntegration instance
    """
    global _global_cultural_integration
    _global_cultural_integration = CulturalIntegration(**kwargs)
    return _global_cultural_integration


# Convenience functions for common operations
def enhance_bias_with_culture(bias_results: Dict[str, Any],
                              sender_culture: str,
                              receiver_culture: str,
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function for enhancing bias results with cultural context.

    Args:
        bias_results: Original bias detection results
        sender_culture: Sender's culture code
        receiver_culture: Receiver's culture code
        context: Optional contextual information

    Returns:
        Culturally-enhanced bias detection results
    """
    integration = get_cultural_integration()
    return integration.enhance_bias_detection(bias_results, sender_culture, receiver_culture, context)


def get_cultural_context(sender_culture: str,
                         receiver_culture: str,
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function for getting cultural context analysis.

    Args:
        sender_culture: Sender's culture code
        receiver_culture: Receiver's culture code
        context: Optional contextual information

    Returns:
        Cultural context analysis
    """
    integration = get_cultural_integration()
    return integration.analyze_cultural_context(sender_culture, receiver_culture, context)