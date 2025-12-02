# BiazNeutralize AI Testing Framework

This comprehensive testing framework ensures the BiazNeutralize AI system meets all functional and non-functional requirements through rigorous validation and quality assurance.

## 📋 Table of Contents

- [Overview](#overview)
- [Test Architecture](#test-architecture)
- [Requirements Validation](#requirements-validation)
- [Test Categories](#test-categories)
- [Running Tests](#running-tests)
- [Quality Gates](#quality-gates)
- [CI/CD Integration](#cicd-integration)
- [Test Data Management](#test-data-management)
- [Performance Benchmarks](#performance-benchmarks)
- [Accessibility Testing](#accessibility-testing)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

The testing framework validates all functional requirements (FR-1 through FR-8) and non-functional requirements (NFR-1 through NFR-6), ensuring the system achieves the defined success criteria (SC-1 through SC-5).

### Key Metrics Targets
- **Bias Detection Accuracy**: F1 score ≥ 0.85 (FR-1)
- **Cultural Appropriateness**: ≥80% expert approval (FR-2)
- **Error-Free Output**: ≥95% success rate (FR-3)
- **Self-Bias Compliance**: 100% prefix requirement (FR-4)
- **Performance**: Response time <3s, throughput >100 req/min (NFR-1, NFR-2)

## 🏗 Test Architecture

```
tests/
├── backend/                    # Backend unit tests
│   ├── test_core_detector.py      # Bias detection algorithm tests
│   ├── test_cultural_adaptation.py # Cultural context tests
│   └── test_llm_integration.py     # LLM integration tests
├── frontend/                   # Frontend component tests
│   ├── components/                 # React component tests
│   └── accessibility/              # Accessibility compliance tests
├── integration/               # API and system integration tests
│   └── test_api_endpoints.py      # End-to-end API testing
├── performance/               # Performance and load tests
│   └── test_benchmarks.py         # System performance benchmarks
├── validation/                # Requirements validation
│   └── test_bias_accuracy.py      # FR-1 through FR-8 validation
├── data/                      # Test data and datasets
│   └── test_datasets.py           # Test data factories and generators
├── e2e/                       # End-to-end workflow tests
│   └── test_bias_analysis_workflow.spec.ts
└── config/                    # Test configuration
    ├── pytest.ini                 # Python test configuration
    └── quality-gates.json         # Quality gate definitions
```

## ✅ Requirements Validation

### Functional Requirements (FR-1 to FR-8)

| Requirement | Test File | Validation Method | Target Metric |
|-------------|-----------|-------------------|---------------|
| **FR-1**: Bias Detection Accuracy | `validation/test_bias_accuracy.py` | Confusion matrix analysis | F1 ≥ 0.85 |
| **FR-2**: Cultural Appropriateness | `validation/test_bias_accuracy.py` | Expert assessment simulation | ≥80% approval |
| **FR-3**: Error-Free Output | `validation/test_bias_accuracy.py` | Exception handling analysis | ≥95% success rate |
| **FR-4**: Self-Bias Compliance | `backend/test_llm_integration.py` | Prefix validation | 100% compliance |
| **FR-5**: Real-Time Processing | `performance/test_benchmarks.py` | Latency measurement | <3s response |
| **FR-6**: Cultural Adaptation | `backend/test_cultural_adaptation.py` | Cross-cultural comparison | Score variation >0.1 |
| **FR-7**: Bias Type Coverage | `validation/test_bias_accuracy.py` | Category detection | 10+ bias types |
| **FR-8**: Neutralization Quality | `integration/test_api_endpoints.py` | Semantic similarity | Quality score ≥0.8 |

### Non-Functional Requirements (NFR-1 to NFR-6)

| Requirement | Test File | Validation Method | Target Metric |
|-------------|-----------|-------------------|---------------|
| **NFR-1**: Response Time | `performance/test_benchmarks.py` | Performance benchmarks | P95 <3s |
| **NFR-2**: Throughput | `performance/test_benchmarks.py` | Load testing | >100 req/min |
| **NFR-3**: Memory Usage | `performance/test_benchmarks.py` | Resource monitoring | <2GB |
| **NFR-4**: Concurrent Users | `performance/test_benchmarks.py` | Concurrency testing | ≥50 users |
| **NFR-5**: Availability | Manual monitoring | Uptime tracking | 99.5% |
| **NFR-6**: Scalability | `performance/test_benchmarks.py` | Stress testing | Linear scaling |

## 🧪 Test Categories

### 1. Unit Tests
- **Backend**: Core bias detection, cultural adaptation, LLM integration
- **Frontend**: React components with React Testing Library
- **Coverage Target**: 80% line coverage

```bash
# Run backend unit tests
pytest tests/backend/ -m unit --cov=src --cov-report=html

# Run frontend unit tests
cd bias-dashboard && npm test
```

### 2. Integration Tests
- **API Endpoints**: Complete request/response workflows
- **Component Integration**: Cross-system communication
- **Database Integration**: Data persistence and retrieval

```bash
# Run integration tests
pytest tests/integration/ -m integration
```

### 3. Performance Tests
- **Load Testing**: System behavior under normal load
- **Stress Testing**: Breaking point identification
- **Benchmark Testing**: Performance regression detection

```bash
# Run performance tests
pytest tests/performance/ -m "slow or performance"
```

### 4. End-to-End Tests
- **User Workflows**: Complete bias analysis journeys
- **Cross-Browser**: Chrome, Firefox, Safari compatibility
- **Mobile Responsive**: Touch and responsive design

```bash
# Run E2E tests
npx playwright test tests/e2e/
```

### 5. Accessibility Tests
- **WCAG 2.1 Compliance**: Level AA standards
- **Screen Reader**: ARIA and semantic HTML
- **Keyboard Navigation**: Full keyboard accessibility

```bash
# Run accessibility tests
npm run test:a11y
```

## 🚀 Running Tests

### Prerequisites

```bash
# Backend dependencies
pip install -r requirements.txt
pip install -r bias-engine/requirements.txt

# Frontend dependencies
cd bias-dashboard && npm install

# E2E test dependencies
npx playwright install
```

### Quick Start

```bash
# Run all tests
make test

# Run specific test categories
make test-unit          # Unit tests only
make test-integration   # Integration tests only
make test-performance   # Performance tests only
make test-e2e          # End-to-end tests only
```

### Detailed Test Commands

#### Backend Tests
```bash
# All backend tests
pytest tests/backend/ -v

# Specific test files
pytest tests/backend/test_core_detector.py -v
pytest tests/validation/test_bias_accuracy.py::TestBiasValidationFramework::test_fr1_bias_detection_accuracy

# With coverage
pytest tests/backend/ --cov=src --cov-report=html --cov-fail-under=80
```

#### Frontend Tests
```bash
# All frontend tests
cd bias-dashboard && npm test

# Watch mode
npm test -- --watch

# With coverage
npm test -- --coverage --watchAll=false
```

#### Performance Tests
```bash
# Quick performance check
pytest tests/performance/ -k "not slow" -v

# Full performance suite
pytest tests/performance/ --tb=short -v

# Specific benchmarks
pytest tests/performance/test_benchmarks.py::TestCoreDetectorPerformance::test_single_text_latency
```

#### E2E Tests
```bash
# All E2E tests
npx playwright test

# Specific browser
npx playwright test --project=chromium

# Headed mode (visible browser)
npx playwright test --headed

# Debug mode
npx playwright test --debug
```

## 🏆 Quality Gates

Quality gates enforce minimum standards before deployment:

### Automated Gates
- **Test Pass Rate**: ≥95% of all tests must pass
- **Code Coverage**: ≥80% line coverage for backend, ≥75% for frontend
- **Performance**: Response time <3s, throughput >100 req/min
- **Security**: No critical or high-severity vulnerabilities
- **Accessibility**: WCAG 2.1 Level AA compliance

### Success Criteria Gates
- **SC-1**: F1 score ≥ 0.85 for bias detection
- **SC-2**: ≥80% cultural appropriateness approval
- **SC-3**: ≥95% error-free output rate
- **SC-4**: 100% self-bias compliance
- **SC-5**: Performance requirements met

### Blocking Conditions
- Any functional requirement (FR-1 to FR-8) not met
- Critical security vulnerabilities
- Performance degradation >20%
- Accessibility violations

## 🔄 CI/CD Integration

### GitHub Actions Pipeline

The automated test pipeline runs on every push and pull request:

1. **Code Quality Checks**
   - Linting (Black, Flake8, ESLint)
   - Type checking (MyPy, TypeScript)
   - Security scanning (Bandit, Safety)

2. **Unit Tests**
   - Backend tests (Python 3.9, 3.10, 3.11)
   - Frontend tests (Node.js 18.x)

3. **Integration Tests**
   - API endpoint testing
   - Database integration
   - Redis integration

4. **Validation Framework**
   - FR-1 to FR-8 validation
   - Success criteria checking
   - Performance benchmarks

5. **E2E Tests** (on main branch)
   - Complete user workflows
   - Cross-browser testing
   - Mobile responsive testing

6. **Deployment Readiness**
   - All quality gates passed
   - Success criteria met
   - Security cleared

### Configuration Files
- **`.github/workflows/test-pipeline.yml`**: Complete CI/CD pipeline
- **`tests/config/pytest.ini`**: Python test configuration
- **`tests/config/quality-gates.json`**: Quality gate definitions

## 📊 Test Data Management

### Test Datasets

The framework includes comprehensive test datasets:

```python
# Generate test datasets
python tests/data/test_datasets.py

# Available datasets
bias_validation_cases.json      # 700 bias test cases
neutral_validation_cases.json   # 200 neutral test cases
cultural_validation_cases.json  # 100 cultural test cases
edge_case_dataset.json          # Edge cases and robustness tests
performance_validation_cases.json # Performance test cases
```

### Data Categories
- **Bias Types**: 10+ cognitive bias categories
- **Cultural Contexts**: 10 different cultural settings
- **Severity Levels**: Low, medium, high, critical
- **Text Lengths**: Short, medium, long, very long
- **Edge Cases**: Empty text, special characters, unicode

### Mock Data Generation
```python
from tests.data.test_datasets import BiasDataFactory

factory = BiasDataFactory()

# Generate specific bias types
confirmation_bias_cases = factory.generate_bias_test_case(
    bias_type=BiasType.CONFIRMATION_BIAS,
    severity=SeverityLevel.HIGH,
    cultural_context=CulturalContext.EN_US
)

# Generate batch test cases
batch_cases = factory.generate_batch_test_cases(
    count=100,
    bias_distribution={
        BiasType.CONFIRMATION_BIAS: 0.3,
        BiasType.ANCHORING_BIAS: 0.2,
        # ... more distributions
    }
)
```

## ⚡ Performance Benchmarks

### Target Metrics
- **Response Time**: <3 seconds for 95% of requests
- **Throughput**: >100 requests per minute
- **Memory Usage**: <2GB under normal load
- **Concurrent Users**: ≥50 simultaneous users
- **Scalability**: Linear performance up to 1000 req/min

### Running Benchmarks
```bash
# Core detector performance
pytest tests/performance/test_benchmarks.py::TestCoreDetectorPerformance -v

# API performance
pytest tests/performance/test_benchmarks.py::TestAPIPerformance -v

# End-to-end performance
pytest tests/performance/test_benchmarks.py::TestEndToEndPerformance -v

# Full performance suite
pytest tests/performance/ -m "slow or performance" --tb=short
```

### Performance Reports
```bash
# Generate performance report
python tests/performance/generate_report.py

# View HTML reports
open tests/reports/performance_report.html
```

## ♿ Accessibility Testing

### WCAG 2.1 Compliance

The framework ensures Level AA compliance across all components:

#### Automated Accessibility Tests
```bash
# Run accessibility tests
npm run test:a11y

# Specific components
npm test tests/frontend/accessibility/test_accessibility.test.tsx
```

#### Manual Testing Guidelines
1. **Keyboard Navigation**: All functionality accessible via keyboard
2. **Screen Reader**: Test with NVDA, JAWS, VoiceOver
3. **Color Contrast**: 4.5:1 ratio for normal text, 3:1 for large text
4. **Focus Management**: Visible focus indicators
5. **ARIA Labels**: Proper semantic markup

#### Accessibility Tools
- **jest-axe**: Automated a11y testing
- **@testing-library/user-event**: Keyboard simulation
- **Lighthouse**: Accessibility audits
- **axe-core**: WCAG validation

## 🐛 Troubleshooting

### Common Issues

#### Test Failures
```bash
# Verbose output for debugging
pytest tests/backend/test_core_detector.py -v -s

# Run only failed tests
pytest --lf

# Skip slow tests during development
pytest -m "not slow"
```

#### Performance Issues
```bash
# Profile test execution
pytest tests/backend/ --durations=10

# Monitor resource usage
pytest tests/performance/ --capture=no

# Debug memory usage
python -m memory_profiler tests/performance/test_benchmarks.py
```

#### E2E Test Issues
```bash
# Run in headed mode to see browser
npx playwright test --headed

# Debug mode with step-by-step execution
npx playwright test --debug

# Trace mode for detailed debugging
npx playwright test --trace on

# View trace files
npx playwright show-trace trace.zip
```

### Environment Setup

#### Backend Environment
```bash
# Virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Dependencies
pip install -r requirements.txt
pip install -r bias-engine/requirements.txt

# Environment variables
cp .env.example .env
# Edit .env with appropriate values
```

#### Frontend Environment
```bash
cd bias-dashboard
npm install

# Environment variables
cp .env.example .env.local
# Edit .env.local with API endpoints
```

#### Test Services
```bash
# Redis for caching
docker run -p 6379:6379 redis:7-alpine

# Test database
docker run -p 5432:5432 -e POSTGRES_PASSWORD=test postgres:14
```

### Test Data Issues
```bash
# Regenerate test datasets
python tests/data/test_datasets.py

# Clear test caches
pytest --cache-clear

# Reset test database
python scripts/reset_test_db.py
```

### CI/CD Issues
```bash
# Run locally with act (GitHub Actions simulator)
act push

# Debug specific workflow steps
act push -j quality-checks

# View workflow logs
gh run view --log
```

## 📈 Test Reporting

### Coverage Reports
```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Generate XML for CI
pytest --cov=src --cov-report=xml

# Check coverage thresholds
pytest --cov=src --cov-fail-under=80
```

### Test Results
```bash
# JUnit XML for CI integration
pytest --junit-xml=test-results.xml

# JSON report
pytest --json-report --json-report-file=test-report.json

# HTML report
pytest --html=test-report.html --self-contained-html
```

### Performance Reports
Performance test results are automatically collected and can be viewed in the CI pipeline artifacts or generated locally:

```bash
python tests/performance/generate_report.py
```

## 🔧 Test Configuration

### Environment Variables
```bash
# Backend testing
TESTING=true
DATABASE_URL=postgresql://test:test@localhost/test_db
REDIS_URL=redis://localhost:6379/1

# Frontend testing
REACT_APP_API_URL=http://localhost:8000
REACT_APP_ENV=test

# E2E testing
BASE_URL=http://localhost:3000
API_URL=http://localhost:8000
HEADLESS=true
```

### Test Markers
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only slow/performance tests
pytest -m slow

# Skip specific markers
pytest -m "not slow and not integration"

# Custom combinations
pytest -m "unit and cultural"
```

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/)
- [Playwright Documentation](https://playwright.dev/)
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [Python Testing Best Practices](https://realpython.com/python-testing/)

## 🤝 Contributing to Tests

### Adding New Tests
1. Follow the existing test structure and naming conventions
2. Include appropriate test markers (`@pytest.mark.unit`, `@pytest.mark.integration`, etc.)
3. Add test documentation and expected outcomes
4. Update this README if adding new test categories
5. Ensure tests are deterministic and can run in parallel

### Test Review Checklist
- [ ] Tests cover both happy path and edge cases
- [ ] Performance implications considered
- [ ] Accessibility requirements verified
- [ ] Cultural context variations tested
- [ ] Error handling validated
- [ ] Documentation updated

---

For questions or issues with the testing framework, please refer to the troubleshooting section above or contact the development team.