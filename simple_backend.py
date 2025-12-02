#!/usr/bin/env python3
"""
Simplified BiazNeutralize AI Backend Server
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import random
import time
from datetime import datetime

app = FastAPI(
    title="BiazNeutralize AI API",
    description="Simplified bias detection and neutralization API",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class AnalysisRequest(BaseModel):
    text: str
    language: str = "auto"
    include_suggestions: bool = True

class BiasDetection(BaseModel):
    type: str
    start_position: int
    end_position: int
    affected_text: str
    level: str
    confidence: float
    description: str
    suggestions: List[str]

class AnalysisResult(BaseModel):
    text: str
    neutralized_text: Optional[str] = None
    overall_bias_score: float
    detections: List[BiasDetection]
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str

# Mock bias detection data for demo
BIAS_PATTERNS = {
    "men are naturally better": {
        "type": "Gender Bias",
        "level": "high",
        "suggestions": [
            "Programming ability varies among individuals regardless of gender",
            "Consider individual skills rather than gender-based assumptions"
        ]
    },
    "women are naturally better": {
        "type": "Gender Bias",
        "level": "high",
        "suggestions": [
            "Avoid gender-based generalizations",
            "Focus on individual capabilities"
        ]
    },
    "young and energetic": {
        "type": "Age Bias",
        "level": "moderate",
        "suggestions": [
            "Use 'motivated and dedicated' instead",
            "Focus on work ethic rather than age-related terms"
        ]
    },
    "naturally better": {
        "type": "Gender Bias",
        "level": "high",
        "suggestions": [
            "Remove assumptions about natural abilities",
            "Focus on skills and experience"
        ]
    },
    "guys": {
        "type": "Gender Bias",
        "level": "low",
        "suggestions": [
            "Use 'team', 'everyone', or 'colleagues'",
            "Use gender-neutral language"
        ]
    },
    "männer": {
        "type": "Gender Bias",
        "level": "moderate",
        "suggestions": [
            "Verwenden Sie geschlechtsneutrale Sprache",
            "Fokussieren Sie auf individuelle Fähigkeiten"
        ]
    },
    "frauen": {
        "type": "Gender Bias",
        "level": "moderate",
        "suggestions": [
            "Verwenden Sie geschlechtsneutrale Sprache",
            "Vermeiden Sie Geschlechterstereotype"
        ]
    },
    "old": {
        "type": "Age Bias",
        "level": "moderate",
        "suggestions": [
            "Use 'experienced' instead of age-related terms",
            "Focus on skills and qualifications"
        ]
    },
    "too old": {
        "type": "Age Bias",
        "level": "high",
        "suggestions": [
            "Remove age-based discrimination",
            "Focus on relevant experience and skills"
        ]
    },
    "he": {
        "type": "Gender Bias",
        "level": "low",
        "suggestions": [
            "Use 'they' for gender-neutral reference",
            "Consider using the person's name"
        ]
    },
    "she": {
        "type": "Gender Bias",
        "level": "low",
        "suggestions": [
            "Use 'they' for gender-neutral reference",
            "Consider using the person's name"
        ]
    }
}

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.post("/api/v1/analyze", response_model=AnalysisResult)
async def analyze_text(request: AnalysisRequest):
    """Analyze text for bias"""
    start_time = time.time()

    # Log the incoming request for debugging
    print(f"🔍 Analyzing text: '{request.text}'")

    text = request.text.lower()
    detections = []
    overall_score = 0.0

    # Mock bias detection
    for pattern, info in BIAS_PATTERNS.items():
        if pattern in text:
            start_pos = text.find(pattern)
            end_pos = start_pos + len(pattern)

            detection = BiasDetection(
                type=info["type"],
                start_position=start_pos,
                end_position=end_pos,
                affected_text=request.text[start_pos:end_pos],
                level=info["level"],
                confidence=random.uniform(0.7, 0.95),
                description=f"Detected {info['type'].lower()} in text",
                suggestions=info["suggestions"]
            )
            detections.append(detection)
            print(f"✅ Found bias pattern: '{pattern}' -> {info['type']}")

            # Increase bias score based on severity
            if info["level"] == "high":
                overall_score += 0.3
            elif info["level"] == "moderate":
                overall_score += 0.2
            else:
                overall_score += 0.1

    # Cap at 1.0
    overall_score = min(overall_score, 1.0)

    print(f"📊 Overall bias score: {overall_score}, Found {len(detections)} detections")

    # Generate neutralized text
    neutralized = request.text
    neutralized = neutralized.replace("Men are naturally better at programming than women",
                                    "Programming ability varies among individuals regardless of gender")
    neutralized = neutralized.replace("young and energetic", "motivated and dedicated")
    neutralized = neutralized.replace("guys", "team")
    neutralized = neutralized.replace("he ", "they ")
    neutralized = neutralized.replace("she ", "they ")

    processing_time = time.time() - start_time

    result = AnalysisResult(
        text=request.text,
        neutralized_text=neutralized if neutralized != request.text else None,
        overall_bias_score=overall_score,
        detections=detections,
        processing_time=processing_time
    )

    print(f"🎯 Returning result with {len(result.detections)} detections")
    return result

@app.get("/api/v1/models")
async def get_models():
    """Get available models"""
    return {
        "available_models": [
            {
                "name": "bias-detector-v1",
                "type": "classification",
                "supported_languages": ["en"],
                "version": "1.0.0"
            }
        ]
    }

@app.get("/api/v1/config")
async def get_config():
    """Get configuration"""
    return {
        "supported_bias_types": [
            "Gender Bias",
            "Age Bias",
            "Racial Bias",
            "Religious Bias",
            "Cultural Bias",
            "Socioeconomic Bias"
        ],
        "supported_languages": ["en"],
        "max_text_length": 10000,
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "BiazNeutralize AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)