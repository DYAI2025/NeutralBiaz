"""
Pydantic models for request/response validation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, validator


class BiasType(str, Enum):
    """Types of bias that can be detected."""
    GENDER = "gender"
    RACIAL = "racial"
    AGE = "age"
    RELIGIOUS = "religious"
    POLITICAL = "political"
    SOCIOECONOMIC = "socioeconomic"
    CULTURAL = "cultural"
    DISABILITY = "disability"
    LGBTQ = "lgbtq"
    UNKNOWN = "unknown"


class BiasLevel(str, Enum):
    """Levels of bias severity."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnalysisMode(str, Enum):
    """Analysis modes for different use cases."""
    FAST = "fast"
    ACCURATE = "accurate"
    COMPREHENSIVE = "comprehensive"


class CulturalProfile(str, Enum):
    """Cultural profiles for context-aware analysis."""
    NEUTRAL = "neutral"
    WESTERN = "western"
    EASTERN = "eastern"
    GLOBAL = "global"


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(description="Service status")
    timestamp: datetime = Field(description="Response timestamp")
    version: str = Field(description="Application version")
    uptime: float = Field(description="Uptime in seconds")
    dependencies: Dict[str, str] = Field(description="Dependency status")


class BiasDetection(BaseModel):
    """Individual bias detection result."""
    type: BiasType = Field(description="Type of bias detected")
    level: BiasLevel = Field(description="Severity level of bias")
    confidence: float = Field(ge=0, le=1, description="Confidence score (0-1)")
    description: str = Field(description="Human-readable description")
    affected_text: str = Field(description="Text span that contains bias")
    start_position: int = Field(ge=0, description="Start position in original text")
    end_position: int = Field(ge=0, description="End position in original text")
    suggestions: List[str] = Field(description="Suggestions for neutralization")

    @validator("end_position")
    def validate_positions(cls, v, values):
        """Validate that end position is after start position."""
        if "start_position" in values and v <= values["start_position"]:
            raise ValueError("end_position must be greater than start_position")
        return v


class AnalysisRequest(BaseModel):
    """Request model for text analysis."""
    text: str = Field(min_length=1, max_length=10000, description="Text to analyze")
    mode: AnalysisMode = Field(default=AnalysisMode.FAST, description="Analysis mode")
    cultural_profile: CulturalProfile = Field(
        default=CulturalProfile.NEUTRAL,
        description="Cultural context for analysis"
    )
    language: Optional[str] = Field(
        default="auto",
        description="Language code (ISO 639-1) or 'auto' for detection"
    )
    bias_types: Optional[List[BiasType]] = Field(
        default=None,
        description="Specific bias types to check (all if not specified)"
    )
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Minimum confidence threshold for reporting bias"
    )
    include_suggestions: bool = Field(
        default=True,
        description="Include neutralization suggestions"
    )

    @validator("text")
    def validate_text(cls, v):
        """Validate text is not empty after stripping."""
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis."""
    texts: List[str] = Field(
        min_items=1,
        max_items=100,
        description="List of texts to analyze"
    )
    mode: AnalysisMode = Field(default=AnalysisMode.FAST, description="Analysis mode")
    cultural_profile: CulturalProfile = Field(
        default=CulturalProfile.NEUTRAL,
        description="Cultural context for analysis"
    )
    language: Optional[str] = Field(
        default="auto",
        description="Language code (ISO 639-1) or 'auto' for detection"
    )
    bias_types: Optional[List[BiasType]] = Field(
        default=None,
        description="Specific bias types to check"
    )
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Minimum confidence threshold"
    )
    include_suggestions: bool = Field(
        default=True,
        description="Include neutralization suggestions"
    )

    @validator("texts")
    def validate_texts(cls, v):
        """Validate all texts are non-empty."""
        for i, text in enumerate(v):
            if not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty")
        return [text.strip() for text in v]


class AnalysisResult(BaseModel):
    """Result model for text analysis."""
    text: str = Field(description="Original text analyzed")
    language: str = Field(description="Detected or specified language")
    overall_bias_score: float = Field(
        ge=0,
        le=1,
        description="Overall bias score (0=no bias, 1=high bias)"
    )
    bias_level: BiasLevel = Field(description="Overall bias level")
    detections: List[BiasDetection] = Field(description="Individual bias detections")
    neutralized_text: Optional[str] = Field(
        default=None,
        description="Suggested neutralized version"
    )
    processing_time: float = Field(ge=0, description="Processing time in seconds")
    model_version: str = Field(description="Version of model used")


class BatchAnalysisResult(BaseModel):
    """Result model for batch analysis."""
    results: List[AnalysisResult] = Field(description="Individual analysis results")
    total_processed: int = Field(ge=0, description="Number of texts processed")
    total_processing_time: float = Field(ge=0, description="Total processing time")
    summary: Dict[str, Union[int, float]] = Field(description="Batch summary statistics")


class ModelInfo(BaseModel):
    """Information about an available model."""
    name: str = Field(description="Model name")
    version: str = Field(description="Model version")
    description: str = Field(description="Model description")
    supported_languages: List[str] = Field(description="Supported language codes")
    supported_bias_types: List[BiasType] = Field(description="Supported bias types")
    accuracy: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Model accuracy score"
    )
    is_default: bool = Field(description="Whether this is the default model")
    loaded: bool = Field(description="Whether model is currently loaded")


class ModelsResponse(BaseModel):
    """Response model for available models."""
    models: List[ModelInfo] = Field(description="Available models")
    default_model: str = Field(description="Name of default model")
    total_models: int = Field(ge=0, description="Total number of models")


class ConfigResponse(BaseModel):
    """Response model for system configuration."""
    app_name: str = Field(description="Application name")
    app_version: str = Field(description="Application version")
    environment: str = Field(description="Environment name")
    supported_languages: List[str] = Field(description="Supported languages")
    supported_bias_types: List[BiasType] = Field(description="Supported bias types")
    cultural_profiles: List[CulturalProfile] = Field(description="Available cultural profiles")
    analysis_modes: List[AnalysisMode] = Field(description="Available analysis modes")
    default_settings: Dict[str, Union[str, float, bool]] = Field(
        description="Default configuration values"
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    error: Dict[str, Union[str, Dict]] = Field(description="Error details")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Input validation failed",
                    "details": {
                        "validation_errors": [
                            {
                                "loc": ["text"],
                                "msg": "field required",
                                "type": "value_error.missing"
                            }
                        ]
                    }
                }
            }
        }