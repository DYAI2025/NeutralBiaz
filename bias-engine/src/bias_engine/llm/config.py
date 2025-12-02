"""
LLM configuration management and environment handling.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from pydantic import BaseSettings, Field, validator
from pydantic_settings import BaseSettings

from .models import LLMProvider, LLMConfig, RateLimitConfig
from ..core.exceptions import BiasEngineError

logger = logging.getLogger(__name__)


class LLMConfigError(BiasEngineError):
    """Base exception for LLM configuration errors."""
    pass


class LLMSettings(BaseSettings):
    """LLM-specific settings loaded from environment variables."""

    # Default LLM provider configuration
    default_provider: LLMProvider = Field(
        default=LLMProvider.ANTHROPIC,
        env="LLM_DEFAULT_PROVIDER",
        description="Default LLM provider"
    )

    # OpenAI configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(default=None, env="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=4096, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.1, env="OPENAI_TEMPERATURE")

    # Anthropic configuration
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    anthropic_base_url: Optional[str] = Field(default=None, env="ANTHROPIC_BASE_URL")
    anthropic_model: str = Field(default="claude-3-sonnet-20240229", env="ANTHROPIC_MODEL")
    anthropic_max_tokens: int = Field(default=4096, env="ANTHROPIC_MAX_TOKENS")
    anthropic_temperature: float = Field(default=0.1, env="ANTHROPIC_TEMPERATURE")

    # Azure OpenAI configuration
    azure_api_key: Optional[str] = Field(default=None, env="AZURE_OPENAI_API_KEY")
    azure_base_url: Optional[str] = Field(default=None, env="AZURE_OPENAI_ENDPOINT")
    azure_deployment: str = Field(default="gpt-4", env="AZURE_OPENAI_DEPLOYMENT")
    azure_max_tokens: int = Field(default=4096, env="AZURE_MAX_TOKENS")
    azure_temperature: float = Field(default=0.1, env="AZURE_TEMPERATURE")

    # Google configuration
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    google_project_id: Optional[str] = Field(default=None, env="GOOGLE_PROJECT_ID")
    google_model: str = Field(default="gemini-pro", env="GOOGLE_MODEL")
    google_max_tokens: int = Field(default=4096, env="GOOGLE_MAX_TOKENS")
    google_temperature: float = Field(default=0.1, env="GOOGLE_TEMPERATURE")

    # Rate limiting configuration
    rate_limit_enabled: bool = Field(default=True, env="LLM_RATE_LIMIT_ENABLED")
    rate_limit_requests_per_minute: int = Field(default=60, env="LLM_RATE_LIMIT_RPM")
    rate_limit_tokens_per_minute: int = Field(default=100000, env="LLM_RATE_LIMIT_TPM")
    rate_limit_burst_size: int = Field(default=10, env="LLM_RATE_LIMIT_BURST")

    # Request configuration
    request_timeout: int = Field(default=60, env="LLM_REQUEST_TIMEOUT")
    max_retries: int = Field(default=3, env="LLM_MAX_RETRIES")
    retry_delay: float = Field(default=1.0, env="LLM_RETRY_DELAY")

    # Prompt configuration
    prompts_file: Optional[str] = Field(default=None, env="LLM_PROMPTS_FILE")
    prompts_cache_enabled: bool = Field(default=True, env="LLM_PROMPTS_CACHE_ENABLED")

    # Quality and validation settings
    response_validation_enabled: bool = Field(default=True, env="LLM_RESPONSE_VALIDATION")
    fallback_enabled: bool = Field(default=True, env="LLM_FALLBACK_ENABLED")
    quality_threshold: float = Field(default=0.7, env="LLM_QUALITY_THRESHOLD")

    # Logging configuration
    log_requests: bool = Field(default=True, env="LLM_LOG_REQUESTS")
    log_responses: bool = Field(default=False, env="LLM_LOG_RESPONSES")  # Privacy consideration
    log_tokens: bool = Field(default=True, env="LLM_LOG_TOKENS")

    @validator("default_provider", pre=True)
    def validate_provider(cls, v):
        """Validate provider string."""
        if isinstance(v, str):
            try:
                return LLMProvider(v.lower())
            except ValueError:
                raise ValueError(f"Invalid LLM provider: {v}")
        return v

    @validator("openai_temperature", "anthropic_temperature", "azure_temperature", "google_temperature")
    def validate_temperature(cls, v):
        """Validate temperature is between 0 and 2."""
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v

    @validator("quality_threshold")
    def validate_quality_threshold(cls, v):
        """Validate quality threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Quality threshold must be between 0 and 1")
        return v

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class LLMConfigManager:
    """
    Manages LLM configurations for different providers and use cases.
    """

    def __init__(self, settings: Optional[LLMSettings] = None):
        """Initialize the configuration manager.

        Args:
            settings: Custom LLM settings (uses default if None)
        """
        self.settings = settings or LLMSettings()
        self._configs: Dict[str, LLMConfig] = {}
        self._load_default_configs()

    def _load_default_configs(self) -> None:
        """Load default configurations for all providers."""
        # OpenAI configuration
        if self.settings.openai_api_key:
            self._configs[LLMProvider.OPENAI.value] = LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key=self.settings.openai_api_key,
                model=self.settings.openai_model,
                base_url=self.settings.openai_base_url,
                max_tokens=self.settings.openai_max_tokens,
                temperature=self.settings.openai_temperature,
                timeout=self.settings.request_timeout,
                max_retries=self.settings.max_retries,
                retry_delay=self.settings.retry_delay
            )

        # Anthropic configuration
        if self.settings.anthropic_api_key:
            self._configs[LLMProvider.ANTHROPIC.value] = LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                api_key=self.settings.anthropic_api_key,
                model=self.settings.anthropic_model,
                base_url=self.settings.anthropic_base_url,
                max_tokens=self.settings.anthropic_max_tokens,
                temperature=self.settings.anthropic_temperature,
                timeout=self.settings.request_timeout,
                max_retries=self.settings.max_retries,
                retry_delay=self.settings.retry_delay
            )

        # Azure OpenAI configuration
        if self.settings.azure_api_key and self.settings.azure_base_url:
            self._configs[LLMProvider.AZURE.value] = LLMConfig(
                provider=LLMProvider.AZURE,
                api_key=self.settings.azure_api_key,
                model=self.settings.azure_deployment,
                base_url=self.settings.azure_base_url,
                max_tokens=self.settings.azure_max_tokens,
                temperature=self.settings.azure_temperature,
                timeout=self.settings.request_timeout,
                max_retries=self.settings.max_retries,
                retry_delay=self.settings.retry_delay
            )

        logger.info(f"Loaded configurations for providers: {list(self._configs.keys())}")

    def get_config(self, provider: Optional[Union[str, LLMProvider]] = None) -> LLMConfig:
        """
        Get LLM configuration for a provider.

        Args:
            provider: LLM provider (uses default if None)

        Returns:
            LLM configuration

        Raises:
            LLMConfigError: If provider not configured
        """
        if provider is None:
            provider = self.settings.default_provider

        if isinstance(provider, str):
            provider_key = provider.lower()
        else:
            provider_key = provider.value

        if provider_key not in self._configs:
            available = list(self._configs.keys())
            raise LLMConfigError(
                f"Provider '{provider_key}' not configured. "
                f"Available providers: {available}"
            )

        return self._configs[provider_key]

    def get_rate_limit_config(self) -> Optional[RateLimitConfig]:
        """
        Get rate limiting configuration.

        Returns:
            Rate limit configuration if enabled, None otherwise
        """
        if not self.settings.rate_limit_enabled:
            return None

        return RateLimitConfig(
            requests_per_minute=self.settings.rate_limit_requests_per_minute,
            tokens_per_minute=self.settings.rate_limit_tokens_per_minute,
            burst_size=self.settings.rate_limit_burst_size
        )

    def get_available_providers(self) -> List[LLMProvider]:
        """
        Get list of configured providers.

        Returns:
            List of available LLM providers
        """
        return [LLMProvider(key) for key in self._configs.keys()]

    def is_provider_available(self, provider: Union[str, LLMProvider]) -> bool:
        """
        Check if a provider is available.

        Args:
            provider: Provider to check

        Returns:
            True if provider is configured and available
        """
        try:
            self.get_config(provider)
            return True
        except LLMConfigError:
            return False

    def add_config(self, config: LLMConfig) -> None:
        """
        Add a custom configuration.

        Args:
            config: LLM configuration to add
        """
        self._configs[config.provider.value] = config
        logger.info(f"Added configuration for provider: {config.provider.value}")

    def remove_config(self, provider: Union[str, LLMProvider]) -> None:
        """
        Remove a provider configuration.

        Args:
            provider: Provider to remove
        """
        if isinstance(provider, LLMProvider):
            provider_key = provider.value
        else:
            provider_key = provider.lower()

        if provider_key in self._configs:
            del self._configs[provider_key]
            logger.info(f"Removed configuration for provider: {provider_key}")

    def validate_configurations(self) -> Dict[str, List[str]]:
        """
        Validate all configurations.

        Returns:
            Dictionary with validation results for each provider
        """
        results = {}

        for provider_key, config in self._configs.items():
            issues = []

            # Check API key
            if not config.api_key or config.api_key == "your-api-key-here":
                issues.append("Invalid or missing API key")

            # Check Azure specific requirements
            if config.provider == LLMProvider.AZURE:
                if not config.base_url:
                    issues.append("Azure requires base_url (endpoint)")

            # Check model name
            if not config.model:
                issues.append("Missing model name")

            # Check timeout and retry settings
            if config.timeout <= 0:
                issues.append("Invalid timeout value")

            if config.max_retries < 0:
                issues.append("Invalid max_retries value")

            results[provider_key] = issues

        return results

    def get_fallback_config(self) -> Optional[LLMConfig]:
        """
        Get fallback configuration if default provider fails.

        Returns:
            Fallback LLM configuration or None
        """
        if not self.settings.fallback_enabled:
            return None

        # Try to find an alternative provider
        default_provider = self.settings.default_provider.value
        available_providers = [k for k in self._configs.keys() if k != default_provider]

        if available_providers:
            fallback_provider = available_providers[0]
            logger.info(f"Using fallback provider: {fallback_provider}")
            return self._configs[fallback_provider]

        return None

    def export_config(self, provider: Union[str, LLMProvider]) -> Dict[str, Any]:
        """
        Export configuration as dictionary (excluding sensitive data).

        Args:
            provider: Provider to export

        Returns:
            Configuration dictionary
        """
        config = self.get_config(provider)

        return {
            "provider": config.provider.value,
            "model": config.model,
            "base_url": config.base_url,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
            "retry_delay": config.retry_delay,
            # API key excluded for security
            "has_api_key": bool(config.api_key)
        }

    def get_usage_statistics(self) -> Dict[str, Any]:
        """
        Get usage statistics for all providers.

        Returns:
            Usage statistics dictionary
        """
        # This would typically integrate with a metrics system
        # For now, return basic configuration info
        return {
            "configured_providers": len(self._configs),
            "default_provider": self.settings.default_provider.value,
            "rate_limiting_enabled": self.settings.rate_limit_enabled,
            "validation_enabled": self.settings.response_validation_enabled,
            "fallback_enabled": self.settings.fallback_enabled,
            "providers": list(self._configs.keys())
        }


# Global configuration manager instance
llm_settings = LLMSettings()
config_manager = LLMConfigManager(llm_settings)