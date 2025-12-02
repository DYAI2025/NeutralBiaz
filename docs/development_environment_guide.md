# BiazNeutralize AI - Development Environment Setup Guide

## 🚀 Quick Start

This guide provides step-by-step instructions for setting up the complete BiazNeutralize AI development environment, ensuring all agents can work efficiently from day one.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows 10+
- **RAM**: Minimum 16GB (32GB recommended for ML models)
- **Storage**: 50GB+ free space
- **Internet**: Stable connection for model downloads and API calls

### Software Dependencies
```bash
# Core tools
- Docker Desktop 20.10+
- Node.js 18+
- Python 3.11+
- Git 2.30+

# Development tools
- VS Code / PyCharm / IntelliJ
- Postman or similar API testing tool
- Terminal/Command Line access
```

## 🛠️ Environment Setup

### 1. Repository Setup
```bash
# Clone the repository
git clone https://github.com/your-org/BiazNeutralize_AI.git
cd BiazNeutralize_AI

# Initialize submodules (for anti-bias framework)
git submodule init
git submodule update

# Create development branches
git checkout -b dev/your-agent-name
```

### 2. Python Backend Environment
```bash
# Navigate to backend directory
cd bias-engine

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLP models
python -m spacy download en_core_web_lg
python -m spacy download de_core_news_lg
python -c "import stanza; stanza.download('en'); stanza.download('de')"

# Download transformer models
python -c "
from transformers import AutoModel, AutoTokenizer
AutoModel.from_pretrained('distilbert-base-multilingual-cased')
AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')
"
```

### 3. React Frontend Environment
```bash
# Navigate to frontend directory
cd ../bias-dashboard

# Install Node.js dependencies
npm install

# Install additional UI libraries
npm install @headlessui/react @heroicons/react chart.js react-chartjs-2

# Verify setup
npm run dev
```

### 4. Docker Environment
```bash
# Return to project root
cd ..

# Build all containers
docker-compose build

# Start development environment
docker-compose up -d

# Verify containers are running
docker-compose ps
```

### 5. Configuration Setup
```bash
# Copy environment templates
cp .env.template .env
cp bias-engine/.env.template bias-engine/.env
cp bias-dashboard/.env.template bias-dashboard/.env

# Edit environment files with your API keys
nano .env
```

### Example .env Configuration
```bash
# .env (root)
COMPOSE_PROJECT_NAME=biaz_neutralize

# bias-engine/.env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
AZURE_OPENAI_API_KEY=your_azure_key_here
ENVIRONMENT=development
LOG_LEVEL=INFO

# Database settings
REDIS_URL=redis://localhost:6379
DATABASE_URL=sqlite:///./bias_analysis.db

# Model paths
SPACY_MODEL_EN=en_core_web_lg
SPACY_MODEL_DE=de_core_news_lg
HUGGINGFACE_CACHE_DIR=./models/huggingface

# bias-dashboard/.env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_ENVIRONMENT=development
REACT_APP_LOG_LEVEL=debug
```

## 📁 Project Structure Overview

```
BiazNeutralize_AI/
├── bias-engine/                 # Python backend
│   ├── src/
│   │   ├── api/                # FastAPI endpoints
│   │   ├── bias/               # Bias detection engine
│   │   ├── cultural/           # Cultural analysis engine
│   │   ├── llm/                # LLM integration
│   │   ├── models/             # Data models
│   │   └── utils/              # Utility functions
│   ├── config/                 # Configuration files
│   │   ├── bias_taxonomy.json  # Intersectional taxonomy
│   │   ├── cultural_profiles/  # Hofstede/GLOBE profiles
│   │   └── prompts.yaml        # LLM prompt templates
│   ├── tests/                  # Test suite
│   ├── requirements.txt        # Python dependencies
│   └── Dockerfile             # Backend container
│
├── bias-dashboard/              # React frontend
│   ├── src/
│   │   ├── components/         # React components
│   │   │   ├── BiasHeatmap.tsx
│   │   │   ├── MarkerExplorer.tsx
│   │   │   ├── CulturalContextPanel.tsx
│   │   │   └── HofstedeRadarChart.tsx
│   │   ├── services/           # API services
│   │   ├── utils/              # Frontend utilities
│   │   └── types/              # TypeScript definitions
│   ├── public/                 # Static assets
│   ├── package.json           # Node.js dependencies
│   └── Dockerfile             # Frontend container
│
├── docs/                       # Documentation
│   ├── unified_execution_strategy.md
│   ├── technical_architecture_blueprint.md
│   ├── implementation_roadmap.md
│   └── development_environment_guide.md
│
├── tests/                      # Integration tests
│   ├── data/                   # Test datasets
│   ├── fixtures/               # Test fixtures
│   └── integration/            # End-to-end tests
│
├── config/                     # Global configuration
├── scripts/                    # Utility scripts
├── docker-compose.yml          # Container orchestration
└── README.md                   # Project overview
```

## 🧪 Development Workflow

### Daily Development Cycle
```bash
# 1. Start development environment
docker-compose up -d

# 2. Activate Python environment
cd bias-engine && source venv/bin/activate

# 3. Start backend in development mode
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# 4. In new terminal, start frontend
cd bias-dashboard && npm run dev

# 5. Run tests before committing
cd bias-engine && python -m pytest tests/
cd bias-dashboard && npm test
```

### Agent-Specific Setup

#### Backend Engineer
```bash
# Additional tools for backend development
pip install black isort flake8 mypy
pip install pytest pytest-cov pytest-asyncio

# Pre-commit hooks
pre-commit install

# Database tools
pip install alembic  # For migrations if using PostgreSQL later
```

#### Frontend Developer
```bash
# Additional tools for frontend development
npm install -g @storybook/cli
npm install -g eslint prettier
npm install --save-dev @testing-library/react @testing-library/jest-dom

# Storybook setup (for component development)
npx storybook init
```

#### LLM Engineer
```bash
# LLM-specific tools
pip install openai anthropic azure-openai
pip install jinja2 pyyaml  # For prompt templates
pip install tiktoken  # Token counting

# Prompt engineering tools
pip install langchain langsmith  # Optional for advanced prompt management
```

#### QA Specialist
```bash
# Testing tools
pip install locust  # Load testing
npm install -g lighthouse  # Performance testing

# Data validation tools
pip install great-expectations pandas
```

## 🔧 Development Tools & Extensions

### VS Code Extensions
```json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.black-formatter",
        "ms-python.isort",
        "ms-python.flake8",
        "bradlc.vscode-tailwindcss",
        "esbenp.prettier-vscode",
        "ms-vscode.vscode-typescript-next",
        "ms-vscode.vscode-json",
        "redhat.vscode-yaml"
    ]
}
```

### Debugging Setup
```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "FastAPI Debug",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": ["src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
            "cwd": "${workspaceFolder}/bias-engine",
            "console": "integratedTerminal"
        },
        {
            "name": "React Debug",
            "type": "node",
            "request": "launch",
            "cwd": "${workspaceFolder}/bias-dashboard",
            "runtimeExecutable": "npm",
            "runtimeArgs": ["run", "dev"]
        }
    ]
}
```

## 📊 Monitoring & Observability

### Local Development Monitoring
```bash
# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Access monitoring dashboards
echo "Grafana: http://localhost:3001 (admin/admin)"
echo "Prometheus: http://localhost:9090"
echo "Redis Insight: http://localhost:8001"
```

### Health Checks
```bash
# Backend health
curl http://localhost:8000/health

# Frontend health
curl http://localhost:3000

# Database health
docker exec -it redis redis-cli ping
```

## 🧪 Testing Setup

### Backend Testing
```bash
# Run all tests
cd bias-engine
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/performance/
```

### Frontend Testing
```bash
# Run all tests
cd bias-dashboard
npm test

# Run with coverage
npm run test:coverage

# Run e2e tests
npm run test:e2e
```

### Integration Testing
```bash
# Full system tests
cd tests
python -m pytest integration/ -v

# Performance tests
locust -f performance/locustfile.py --host=http://localhost:8000
```

## 🚀 Quick Validation

### Verify Complete Setup
```bash
# Run validation script
./scripts/validate_environment.sh

# Expected output:
# ✓ Docker containers running
# ✓ Backend API responding
# ✓ Frontend serving
# ✓ Database connected
# ✓ NLP models loaded
# ✓ LLM providers accessible
# ✓ Tests passing
```

### Sample API Test
```bash
# Test basic bias detection
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a test sentence for bias detection.",
    "sender_culture": "de",
    "receiver_culture": "jp"
  }'
```

## 🔍 Troubleshooting

### Common Issues

**Docker Issues**
```bash
# Reset Docker environment
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

**Python Model Issues**
```bash
# Reinstall models
python -m spacy download en_core_web_lg --force
python -m spacy download de_core_news_lg --force
```

**Node.js Issues**
```bash
# Clear npm cache
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

**API Connection Issues**
```bash
# Check API keys
grep -v "^#" bias-engine/.env | grep API_KEY

# Test connectivity
curl -v http://localhost:8000/health
```

### Performance Issues
```bash
# Monitor resource usage
docker stats

# Check logs
docker-compose logs backend
docker-compose logs frontend

# Profile Python code
python -m cProfile -s cumulative src/main.py
```

## 📚 Additional Resources

### Documentation Links
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [spaCy Documentation](https://spacy.io/usage)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

### Team Communication
- **Slack Channel**: #biaz-neutralize-dev
- **Daily Standups**: 9:00 AM CET
- **Code Reviews**: GitHub Pull Requests
- **Emergency Contact**: [Technical Lead Contact]

This development environment guide ensures all team members can quickly set up a consistent, functional development environment and start contributing to the BiazNeutralize AI project immediately.