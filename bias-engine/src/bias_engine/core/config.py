"""
Configuration management for the Bias Engine.
"""

from typing import List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application settings
    app_name: str = Field(default="Bias Engine API", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")

    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")

    # Security settings
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    allowed_hosts: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )

    # Redis settings
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")

    # Model settings
    default_model: str = Field(default="distilbert-base-uncased", env="DEFAULT_MODEL")
    models_cache_dir: str = Field(default="./models", env="MODELS_CACHE_DIR")
    max_model_cache_size: int = Field(default=5, env="MAX_MODEL_CACHE_SIZE")

    # API settings
    max_request_size: int = Field(default=10485760, env="MAX_REQUEST_SIZE")  # 10MB
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")  # 30 seconds
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # 1 hour

    # Bias detection settings
    bias_threshold: float = Field(default=0.7, env="BIAS_THRESHOLD")
    confidence_threshold: float = Field(default=0.8, env="CONFIDENCE_THRESHOLD")
    supported_languages: List[str] = Field(
        default=["en", "es", "fr", "de"],
        env="SUPPORTED_LANGUAGES"
    )

    # Cultural profiles settings
    cultural_profiles_enabled: bool = Field(default=True, env="CULTURAL_PROFILES_ENABLED")
    default_cultural_profile: str = Field(default="neutral", env="DEFAULT_CULTURAL_PROFILE")

    # External APIs - Legacy (kept for backward compatibility)
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")

    # LLM Integration Settings
    llm_enabled: bool = Field(default=True, env="LLM_ENABLED")
    llm_default_provider: str = Field(default="anthropic", env="LLM_DEFAULT_PROVIDER")
    llm_fallback_enabled: bool = Field(default=True, env="LLM_FALLBACK_ENABLED")
    llm_response_validation: bool = Field(default=True, env="LLM_RESPONSE_VALIDATION")
    llm_quality_threshold: float = Field(default=0.7, env="LLM_QUALITY_THRESHOLD")

    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json or text
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")

    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string if needed."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @validator("allowed_hosts", pre=True)
    def parse_allowed_hosts(cls, v):
        """Parse allowed hosts from string if needed."""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v

    @validator("supported_languages", pre=True)
    def parse_supported_languages(cls, v):
        """Parse supported languages from string if needed."""
        if isinstance(v, str):
            return [lang.strip() for lang in v.split(",")]
        return v

    @validator("bias_threshold")
    def validate_bias_threshold(cls, v):
        """Validate bias threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("bias_threshold must be between 0 and 1")
        return v

    @validator("confidence_threshold")
    def validate_confidence_threshold(cls, v):
        """Validate confidence threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        return v

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()