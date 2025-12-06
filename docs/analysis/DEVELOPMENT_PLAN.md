# NeutraBiaz - Iterativer Entwicklungsplan

**Version:** 1.0
**Datum:** 2025-12-06
**Status:** In Planung

---

## 🎯 Entwicklungsziele

### Hauptziele

1. **Integration der vollständigen Detection Engine** in FastAPI
2. **Entfernung aller Demo-Dateien** aus dem Repository
3. **Datenbank-Integration** für Persistenz
4. **LLM-Integration** aktivieren
5. **Production-Ready** Status erreichen

---

## 📋 Entwicklungsphasen-Übersicht

```
Phase 0: Cleanup & Preparation         [1-2 Tage]   🔴 KRITISCH
Phase 1: Core Engine Integration      [3-5 Tage]   🔴 KRITISCH
Phase 2: Database Layer               [2-3 Tage]   🟡 WICHTIG
Phase 3: LLM Integration              [2-3 Tage]   🟡 WICHTIG
Phase 4: Advanced Features            [3-4 Tage]   🟢 OPTIONAL
Phase 5: Production Hardening         [2-3 Tage]   🟡 WICHTIG
Phase 6: Performance Optimization     [2-3 Tage]   🟢 OPTIONAL

Total: 15-23 Arbeitstage (3-5 Wochen)
```

---

## 🚀 Phase 0: Cleanup & Preparation (1-2 Tage)

**Priorität:** 🔴 KRITISCH
**Abhängigkeiten:** Keine
**Ziel:** Codebase bereinigen, klare Struktur schaffen

### Tasks

#### Task 0.1: Demo-Dateien entfernen
**Aufwand:** 0.5 Tage
**Dateien zu löschen:**
```bash
# Zu entfernen
bias-dashboard/demo.html                          # 290 Zeilen
simple_backend.py                                 # 267 Zeilen
scripts/demo_bias_detection.py                    # ~200 Zeilen
scripts/simple_test.py                            # ~100 Zeilen
examples/cultural_adaptation_examples.py          # ~150 Zeilen (optional behalten)

# Zu prüfen
examples/env.llm.example                          # BEHALTEN als Template
```

**Acceptance Criteria:**
- ✅ Alle Demo-HTML-Dateien gelöscht
- ✅ `simple_backend.py` entfernt
- ✅ Demo-Scripts aus `scripts/` entfernt
- ✅ Git commit mit "chore: Remove demo files"

#### Task 0.2: Backend-Architektur konsolidieren
**Aufwand:** 0.5 Tage
**Ziel:** Entscheidung über `src/` vs. `bias-engine/`

**Option A: Migration (EMPFOHLEN)**
```
src/bias_engine/           →  bias-engine/src/bias_engine/detection/
├── core_detector.py       →  ├── core_detector.py
├── rule_based_detector.py →  ├── rule_based_detector.py
├── ml_classifier.py       →  ├── ml_classifier.py
├── nlp_pipeline.py        →  ├── nlp_pipeline.py
├── scoring_algorithms.py  →  ├── scoring_algorithms.py
├── taxonomy_loader.py     →  ├── taxonomy_loader.py
└── config_manager.py      →  └── config_manager.py

src/models/                →  bias-engine/src/bias_engine/models/
└── bias_models.py         →  └── bias_models.py (merge mit schemas.py)
```

**Option B: Symlink (QUICK FIX)**
```python
# In bias-engine/src/bias_engine/
from src.bias_engine import core_detector  # Import from legacy
```

**Empfehlung:** Option A (Migration) für langfristige Wartbarkeit

**Acceptance Criteria:**
- ✅ Klare Entscheidung dokumentiert
- ✅ Entweder Migration durchgeführt ODER Symlinks erstellt
- ✅ Imports in allen Dateien aktualisiert
- ✅ Tests laufen weiterhin

#### Task 0.3: Projekt-Struktur dokumentieren
**Aufwand:** 0.5 Tage

**Zu erstellen:**
```
docs/architecture/
├── ARCHITECTURE.md          # Gesamt-Architektur
├── BACKEND_STRUCTURE.md     # Backend-Organisation
├── FRONTEND_STRUCTURE.md    # Frontend-Organisation
└── DATA_FLOW.md             # Datenfluss-Diagramme
```

**Acceptance Criteria:**
- ✅ Architektur-Diagramme erstellt
- ✅ Datenfluss dokumentiert
- ✅ Komponenten-Interaktionen beschrieben
- ✅ In README.md verlinkt

---

## 🔧 Phase 1: Core Engine Integration (3-5 Tage)

**Priorität:** 🔴 KRITISCH
**Abhängigkeiten:** Phase 0 abgeschlossen
**Ziel:** Echte Bias-Detection in API-Endpoints integrieren

### Tasks

#### Task 1.1: Detection Service Layer erstellen
**Aufwand:** 1 Tag

**Neue Datei:** `bias-engine/src/bias_engine/services/detection_service.py`

```python
"""
Detection Service
Koordiniert alle Detection-Komponenten
"""
from typing import List, Optional
from bias_engine.models.schemas import (
    AnalysisRequest, AnalysisResult, BiasDetection
)
from bias_engine.detection.core_detector import BiasDetectionEngine
from bias_engine.cultural.integration import CulturalIntegration
from bias_engine.llm.pipeline import LLMPipeline

class BiasDetectionService:
    """Main service for bias detection"""

    def __init__(self):
        self.detector = BiasDetectionEngine()
        self.cultural = CulturalIntegration()
        self.llm = None  # Optional LLM

    async def analyze_text(
        self,
        request: AnalysisRequest
    ) -> AnalysisResult:
        """
        Analyze text using full detection pipeline
        """
        # 1. Language detection
        # 2. Core detection (rule + ML)
        # 3. Cultural adaptation
        # 4. Scoring
        # 5. Neutralization (optional)
        pass

    async def analyze_batch(
        self,
        requests: List[AnalysisRequest]
    ) -> List[AnalysisResult]:
        """Batch analysis with optimization"""
        pass
```

**Acceptance Criteria:**
- ✅ `BiasDetectionService` class erstellt
- ✅ Verwendet Core Detector aus Phase 0
- ✅ Methoden für single + batch analysis
- ✅ Unit-Tests geschrieben
- ✅ Logging integriert

#### Task 1.2: Analyze Routes refactoren
**Aufwand:** 1 Tag

**Datei:** `bias-engine/src/bias_engine/api/routes/analyze.py`

**Änderungen:**
```python
# ALT (Mock)
bias_keywords = {
    "he should": BiasType.GENDER,
    # ... hardcoded keywords
}

# NEU (Real Detection)
from bias_engine.services.detection_service import BiasDetectionService

detection_service = BiasDetectionService()

async def perform_bias_analysis(
    text: str,
    request: AnalysisRequest
) -> AnalysisResult:
    """Real bias analysis using full engine"""
    return await detection_service.analyze_text(request)
```

**Acceptance Criteria:**
- ✅ Mock-Code entfernt
- ✅ `BiasDetectionService` integriert
- ✅ Alle Request-Parameter verwendet (mode, cultural_profile, etc.)
- ✅ Error Handling beibehalten
- ✅ Logging erweitert

#### Task 1.3: Model Loading & Caching
**Aufwand:** 1 Tag

**Ziel:** ML-Modelle beim Startup laden, cachen

**Neue Datei:** `bias-engine/src/bias_engine/services/model_manager.py`

```python
"""
Model Manager
Handles ML model loading and caching
"""
import logging
from pathlib import Path
from typing import Dict, Any

class ModelManager:
    """Manages ML models lifecycle"""

    def __init__(self):
        self._models = {}
        self._loaded = False

    async def load_models(self):
        """Load all ML models on startup"""
        # Load spaCy models
        # Load BERT models
        # Load fasttext models
        # Load custom models
        pass

    def get_model(self, name: str) -> Any:
        """Get cached model by name"""
        pass
```

**Integration in `main.py`:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Loading ML models...")
    model_manager = ModelManager()
    await model_manager.load_models()
    app.state.models = model_manager

    yield

    # Shutdown
    logger.info("Unloading models...")
```

**Acceptance Criteria:**
- ✅ `ModelManager` erstellt
- ✅ Modelle beim Startup geladen
- ✅ Lazy Loading für optionale Modelle
- ✅ Memory Monitoring
- ✅ Health Check zeigt Modell-Status

#### Task 1.4: Integration Testing
**Aufwand:** 1 Tag

**Tests erweitern:**
```python
# tests/integration/test_real_detection.py
def test_analyze_with_real_engine():
    """Test API with real detection engine"""
    text = "Men are naturally better at programming"
    response = client.post("/api/v1/analyze", json={
        "text": text,
        "mode": "comprehensive"
    })

    assert response.status_code == 200
    result = response.json()

    # Should detect gender bias
    assert len(result["detections"]) > 0
    assert any(d["type"] == "gender" for d in result["detections"])

    # Should have high confidence
    assert result["overall_bias_score"] > 0.7
```

**Acceptance Criteria:**
- ✅ Integration-Tests für reale Detection
- ✅ Tests für alle Bias-Typen
- ✅ Tests für Cultural Profiles
- ✅ Tests für alle Analysis Modes
- ✅ Performance-Tests (< 2s pro Text)

#### Task 1.5: Batch Processing Optimization
**Aufwand:** 1 Tag

**Ziel:** Parallele Verarbeitung für Batch-Requests

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def analyze_batch(request: BatchAnalysisRequest):
    """Optimized batch processing"""

    # Option 1: Asyncio gather
    tasks = [
        detection_service.analyze_text(text)
        for text in request.texts
    ]
    results = await asyncio.gather(*tasks)

    # Option 2: Thread pool for CPU-bound tasks
    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     results = await loop.run_in_executor(...)
```

**Acceptance Criteria:**
- ✅ Parallele Verarbeitung implementiert
- ✅ Konfigurierbare Worker-Anzahl
- ✅ Memory Limits eingehalten
- ✅ 3-5x Speedup vs. sequential
- ✅ Batch-Performance-Tests

---

## 💾 Phase 2: Database Layer (2-3 Tage)

**Priorität:** 🟡 WICHTIG
**Abhängigkeiten:** Phase 1 abgeschlossen
**Ziel:** Persistenz für Analysen, History, Users

### Tasks

#### Task 2.1: Database Models
**Aufwand:** 1 Tag

**Neue Datei:** `bias-engine/src/bias_engine/models/database.py`

```python
"""
SQLAlchemy Database Models
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    analyses = relationship("Analysis", back_populates="user")

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    text = Column(String, nullable=False)
    language = Column(String, nullable=False)
    overall_bias_score = Column(Float, nullable=False)
    detections = Column(JSON)  # Stored as JSON
    neutralized_text = Column(String, nullable=True)
    processing_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="analyses")

class BiasPattern(Base):
    __tablename__ = "bias_patterns"

    id = Column(Integer, primary_key=True)
    pattern = Column(String, unique=True, nullable=False)
    bias_type = Column(String, nullable=False)
    severity = Column(Float, nullable=False)
    detected_count = Column(Integer, default=0)
    last_seen = Column(DateTime, default=datetime.utcnow)
```

**Acceptance Criteria:**
- ✅ User, Analysis, BiasPattern Models
- ✅ Relationships definiert
- ✅ Indexes für Performance
- ✅ Alembic Migrations erstellt
- ✅ Tests für Models

#### Task 2.2: Database Service
**Aufwand:** 1 Tag

**Neue Datei:** `bias-engine/src/bias_engine/services/database_service.py`

```python
"""
Database Service
Handles all database operations
"""
from typing import List, Optional
from sqlalchemy.orm import Session
from bias_engine.models.database import User, Analysis, BiasPattern
from bias_engine.models.schemas import AnalysisResult

class DatabaseService:
    """Database operations service"""

    def __init__(self, session: Session):
        self.session = session

    async def save_analysis(
        self,
        result: AnalysisResult,
        user_id: Optional[int] = None
    ) -> Analysis:
        """Save analysis result to database"""
        analysis = Analysis(
            user_id=user_id,
            text=result.text,
            language=result.language,
            overall_bias_score=result.overall_bias_score,
            detections=[d.dict() for d in result.detections],
            neutralized_text=result.neutralized_text,
            processing_time=result.processing_time
        )
        self.session.add(analysis)
        await self.session.commit()
        return analysis

    async def get_user_history(
        self,
        user_id: int,
        limit: int = 50,
        offset: int = 0
    ) -> List[Analysis]:
        """Get analysis history for user"""
        return self.session.query(Analysis)\
            .filter(Analysis.user_id == user_id)\
            .order_by(Analysis.created_at.desc())\
            .limit(limit)\
            .offset(offset)\
            .all()
```

**Acceptance Criteria:**
- ✅ CRUD operations für alle Models
- ✅ Pagination support
- ✅ Filtering & Sorting
- ✅ Transaction management
- ✅ Unit-Tests für Service

#### Task 2.3: API Integration
**Aufwand:** 0.5 Tage

**Änderungen in `analyze.py`:**
```python
@router.post("/analyze")
async def analyze_text(
    request: AnalysisRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    # Perform analysis
    result = await detection_service.analyze_text(request)

    # Save to database
    db_service = DatabaseService(db)
    await db_service.save_analysis(result, current_user.id if current_user else None)

    return result
```

**Neue Endpoints:**
```python
@router.get("/history")
async def get_analysis_history(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get user's analysis history"""
    db_service = DatabaseService(db)
    return await db_service.get_user_history(
        current_user.id, limit, offset
    )

@router.get("/analysis/{analysis_id}")
async def get_analysis(
    analysis_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get specific analysis by ID"""
    db_service = DatabaseService(db)
    return await db_service.get_analysis(analysis_id, current_user.id)
```

**Acceptance Criteria:**
- ✅ Database dependency injection
- ✅ Auto-save nach Analysis
- ✅ History Endpoints
- ✅ Analysis retrieval
- ✅ Integration-Tests

---

## 🤖 Phase 3: LLM Integration (2-3 Tage)

**Priorität:** 🟡 WICHTIG
**Abhängigkeiten:** Phase 1 abgeschlossen
**Ziel:** LLM-basierte Debiasing aktivieren

### Tasks

#### Task 3.1: LLM Service Integration
**Aufwand:** 1 Tag

**Datei:** `bias-engine/src/bias_engine/services/llm_service.py`

```python
"""
LLM Service
Coordinates LLM-based bias detection and neutralization
"""
from bias_engine.llm.client import LLMClient
from bias_engine.llm.pipeline import LLMPipeline
from bias_engine.llm.prompts import BiasDetectionPrompts

class LLMService:
    """LLM operations service"""

    def __init__(self):
        self.client = LLMClient()
        self.pipeline = LLMPipeline(self.client)
        self.prompts = BiasDetectionPrompts()

    async def detect_bias_llm(self, text: str) -> List[BiasDetection]:
        """Detect bias using LLM"""
        prompt = self.prompts.get_detection_prompt(text)
        response = await self.client.complete(prompt)
        return self.pipeline.parse_bias_response(response)

    async def neutralize_text(self, text: str, detections: List[BiasDetection]) -> str:
        """Generate neutralized version using LLM"""
        prompt = self.prompts.get_neutralization_prompt(text, detections)
        response = await self.client.complete(prompt)
        return response.content
```

**Acceptance Criteria:**
- ✅ `LLMService` erstellt
- ✅ OpenAI + Anthropic Support
- ✅ Async operations
- ✅ Error handling & retries
- ✅ Rate limiting

#### Task 3.2: Debias Endpoint implementieren
**Aufwand:** 0.5 Tage

**Datei:** `bias-engine/src/bias_engine/api/routes/llm_debiasing.py`

```python
from bias_engine.services.llm_service import LLMService

llm_service = LLMService()

@router.post("/debias")
async def debias_text(request: DebiasRequest) -> DebiasResult:
    """
    Debias text using LLM
    """
    # 1. Detect bias (optional, can use existing detections)
    if not request.detections:
        detections = await llm_service.detect_bias_llm(request.text)
    else:
        detections = request.detections

    # 2. Generate neutralized version
    neutralized = await llm_service.neutralize_text(request.text, detections)

    return DebiasResult(
        original_text=request.text,
        neutralized_text=neutralized,
        detections=detections,
        model=llm_service.client.model
    )
```

**Acceptance Criteria:**
- ✅ Endpoint funktioniert
- ✅ Unterstützt multiple LLM Providers
- ✅ Error handling
- ✅ API-Tests
- ✅ Documentation

#### Task 3.3: Hybrid Detection Mode
**Aufwand:** 1 Tag

**Ziel:** Rule + ML + LLM kombinieren

```python
class BiasDetectionService:
    def __init__(self):
        self.detector = BiasDetectionEngine()  # Rule + ML
        self.llm_service = LLMService()  # LLM

    async def analyze_text(self, request: AnalysisRequest) -> AnalysisResult:
        if request.mode == "fast":
            # Rule-based only
            return await self.detector.detect_bias(request.text)

        elif request.mode == "accurate":
            # Rule + ML
            return await self.detector.detect_bias_with_ml(request.text)

        elif request.mode == "comprehensive":
            # Rule + ML + LLM (hybrid)
            rule_ml_results = await self.detector.detect_bias_with_ml(request.text)
            llm_results = await self.llm_service.detect_bias_llm(request.text)

            # Merge and deduplicate
            return self._merge_detections(rule_ml_results, llm_results)
```

**Acceptance Criteria:**
- ✅ 3 Modi implementiert (fast, accurate, comprehensive)
- ✅ Detection-Merging ohne Duplikate
- ✅ Confidence-Scoring für merged results
- ✅ Performance-Tests für alle Modi
- ✅ Documentation

---

## ⚡ Phase 4: Advanced Features (3-4 Tage)

**Priorität:** 🟢 OPTIONAL
**Abhängigkeiten:** Phasen 1-3 abgeschlossen

### Tasks

#### Task 4.1: Authentifizierung
**Aufwand:** 1.5 Tage

**Zu implementieren:**
- JWT Token Authentication
- User Registration/Login
- API Key Management
- Rate Limiting pro User

#### Task 4.2: Real-time Streaming
**Aufwand:** 1 Tag

**Endpoint:** `POST /api/v1/analyze/stream`

```python
@router.post("/analyze/stream")
async def analyze_stream(request: AnalysisRequest):
    """Stream analysis results as they're detected"""
    async def generate():
        # Stream detections as they're found
        async for detection in detection_service.detect_streaming(request.text):
            yield json.dumps(detection.dict()) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")
```

#### Task 4.3: Analytics Dashboard
**Aufwand:** 1.5 Tage

**Neue Endpoints:**
```python
@router.get("/analytics/overview")
async def get_analytics_overview():
    """Get bias detection statistics"""
    return {
        "total_analyses": 1234,
        "most_common_biases": [...],
        "average_bias_score": 0.45,
        "bias_trends": [...]
    }
```

**Frontend:** Analytics page in React Dashboard

---

## 🛡️ Phase 5: Production Hardening (2-3 Tage)

**Priorität:** 🟡 WICHTIG
**Abhängigkeiten:** Phase 1-2 abgeschlossen

### Tasks

#### Task 5.1: Security Hardening
**Aufwand:** 1 Tag

- Input Validation verschärfen
- SQL Injection Prevention
- XSS Protection
- CORS korrekt konfigurieren
- Security Headers hinzufügen
- API Rate Limiting
- DDoS Protection

#### Task 5.2: Error Handling & Monitoring
**Aufwand:** 1 Tag

- Sentry Integration für Error Tracking
- Structured Logging erweitern
- Health Checks verbessern
- Prometheus Metrics
- Alerting Setup

#### Task 5.3: Documentation
**Aufwand:** 1 Tag

- OpenAPI Spec vervollständigen
- API Usage Examples
- Deployment Guide
- Troubleshooting Guide
- FAQ erstellen

---

## 🚀 Phase 6: Performance Optimization (2-3 Tage)

**Priorität:** 🟢 OPTIONAL
**Abhängigkeiten:** Alle vorherigen Phasen

### Tasks

#### Task 6.1: Caching Layer
**Aufwand:** 1 Tag

- Redis für Result Caching
- Model Caching optimieren
- Query Result Caching

#### Task 6.2: Performance Profiling
**Aufwand:** 1 Tag

- CPU Profiling
- Memory Profiling
- Database Query Optimization
- Bottleneck Identification

#### Task 6.3: Load Testing
**Aufwand:** 1 Tag

- Locust Setup
- Load Test Scenarios
- Capacity Planning
- Auto-scaling Configuration

---

## 📅 Zeitplan & Meilensteine

### Sprint 1 (Woche 1)
- ✅ Phase 0: Cleanup (2 Tage)
- ✅ Phase 1: Start Core Integration (3 Tage)

**Meilenstein:** Demo-frei, Core Engine teilweise integriert

### Sprint 2 (Woche 2)
- ✅ Phase 1: Abschluss Core Integration (2 Tage)
- ✅ Phase 2: Database Layer (3 Tage)

**Meilenstein:** Reale Detection funktioniert, Persistenz vorhanden

### Sprint 3 (Woche 3)
- ✅ Phase 3: LLM Integration (3 Tage)
- ✅ Phase 5: Security Start (2 Tage)

**Meilenstein:** LLM-basierte Debiasing verfügbar

### Sprint 4 (Woche 4-5)
- ⚠️ Phase 4: Advanced Features (optional)
- ✅ Phase 5: Production Hardening Abschluss
- ⚠️ Phase 6: Performance Optimization (optional)

**Meilenstein:** Production-Ready System

---

## 🎯 Prioritäten-Matrix

### Must-Have (Kritisch für Production)
1. Demo-Entfernung (Phase 0)
2. Core Engine Integration (Phase 1)
3. Database Layer (Phase 2)
4. Security Hardening (Phase 5.1)
5. Error Handling (Phase 5.2)

### Should-Have (Wichtig für Features)
6. LLM Integration (Phase 3)
7. Documentation (Phase 5.3)
8. Monitoring (Phase 5.2)

### Nice-to-Have (Optional)
9. Authentication (Phase 4.1)
10. Streaming (Phase 4.2)
11. Analytics (Phase 4.3)
12. Caching (Phase 6.1)
13. Performance Optimization (Phase 6.2-6.3)

---

## 📊 Risiken & Mitigation

### Risiko 1: ML Model Loading Performance
**Wahrscheinlichkeit:** Hoch
**Impact:** Medium
**Mitigation:**
- Lazy Loading für optionale Modelle
- Model Quantization
- Startup-Zeit Monitoring

### Risiko 2: LLM API Kosten
**Wahrscheinlichkeit:** Medium
**Impact:** Hoch
**Mitigation:**
- Caching von LLM Responses
- Rate Limiting
- Fallback zu Rule-based

### Risiko 3: Database Migration Komplexität
**Wahrscheinlichkeit:** Low
**Impact:** Medium
**Mitigation:**
- Alembic Migrations
- Rollback-Strategie
- Staging Environment Testing

---

## ✅ Definition of Done

### Für jede Phase:

1. **Code**
   - ✅ Alle Features implementiert
   - ✅ Code Review durchgeführt
   - ✅ Keine kritischen TODOs

2. **Tests**
   - ✅ Unit-Tests geschrieben (80%+ Coverage)
   - ✅ Integration-Tests vorhanden
   - ✅ Alle Tests grün

3. **Documentation**
   - ✅ API Docs aktualisiert
   - ✅ Code Comments vorhanden
   - ✅ README.md erweitert

4. **Deployment**
   - ✅ Docker Image builds
   - ✅ Kubernetes manifests aktualisiert
   - ✅ Staging-Deployment erfolgreich

---

## 🔄 Iterative Development Process

### Für jeden Task:

1. **Planning**
   - Requirements klären
   - Design-Entscheidungen treffen
   - Acceptance Criteria definieren

2. **Implementation**
   - Test-First Development (TDD)
   - Incremental Commits
   - Code Review nach jedem Feature

3. **Testing**
   - Unit-Tests schreiben
   - Integration-Tests erweitern
   - Manual Testing durchführen

4. **Review & Refactor**
   - Code Review
   - Refactoring für Code Quality
   - Performance-Check

5. **Documentation**
   - Inline Comments
   - API Docs aktualisieren
   - User Guide erweitern

6. **Deployment**
   - Staging Deploy
   - Smoke Tests
   - Production Deploy (nach Approval)

---

## 📈 Success Metrics

### Phase 1 Success:
- ✅ API nutzt Core Detector (nicht Mock)
- ✅ Detection Accuracy > 85%
- ✅ Response Time < 2s pro Text
- ✅ Alle Tests grün

### Phase 2 Success:
- ✅ Alle Analysen in DB gespeichert
- ✅ History Endpoint funktioniert
- ✅ Database Migrations funktionieren
- ✅ Query Performance < 100ms

### Phase 3 Success:
- ✅ LLM Integration funktioniert
- ✅ Neutralized Text Quality > 90%
- ✅ LLM Fallback bei Errors
- ✅ Cost < $0.01 pro Analysis

### Production-Ready Success:
- ✅ 99% Uptime
- ✅ < 2s Response Time (p95)
- ✅ 1000+ req/min Capacity
- ✅ Zero Critical Vulnerabilities

---

**Ende des Entwicklungsplans**
**Nächste Schritte:** Phase 0 starten - Demo-Dateien entfernen
