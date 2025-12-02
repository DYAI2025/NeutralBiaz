# Bias Engine - Backend API

AI-powered bias detection and neutralization engine built with FastAPI.

## 🚀 Features

- **Advanced Bias Detection**: Detect various types of bias in text including gender, racial, age, religious, and cultural bias
- **Real-time Analysis**: Fast text analysis with configurable accuracy modes
- **Batch Processing**: Process multiple texts efficiently
- **Cultural Awareness**: Context-aware analysis with cultural profile support
- **RESTful API**: Well-documented REST API with OpenAPI/Swagger integration
- **Robust Architecture**: Production-ready with comprehensive error handling and logging
- **Containerized**: Docker support for easy deployment

## 🏗️ Project Structure

```
bias-engine/
├── src/
│   └── bias_engine/
│       ├── api/
│       │   └── routes/          # API endpoints
│       ├── core/                # Core configuration and utilities
│       ├── models/              # Data models and schemas
│       ├── services/            # Business logic services
│       └── utils/               # Utility functions
├── tests/                       # Test suite
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── config/                     # Configuration files
├── scripts/                    # Utility scripts
├── requirements.txt            # Dependencies
├── pyproject.toml             # Project configuration
├── Dockerfile                 # Container configuration
└── docker-compose.yml        # Multi-service setup
```

## 🔧 Installation

### Prerequisites

- Python 3.9+
- pip
- (Optional) Docker and Docker Compose

### Local Development Setup

1. **Clone and navigate to the project:**
   ```bash
   cd bias-engine
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the development server:**
   ```bash
   ./scripts/start.sh
   # Or manually:
   PYTHONPATH=src uvicorn bias_engine.main:app --reload
   ```

### Docker Setup

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

2. **Or run with Docker only:**
   ```bash
   docker build -t bias-engine .
   docker run -p 8000:8000 bias-engine
   ```

## 📚 API Endpoints

### Health Checks
- `GET /api/v1/health` - Service health status
- `GET /api/v1/health/ready` - Readiness check
- `GET /api/v1/health/live` - Liveness check

### Text Analysis
- `POST /api/v1/analyze` - Analyze single text for bias
- `POST /api/v1/analyze/batch` - Batch analyze multiple texts

### Configuration
- `GET /api/v1/config` - Get system configuration
- `GET /api/v1/models` - List available models
- `GET /api/v1/models/{model_name}` - Get specific model info

## 🔍 Usage Examples

### Single Text Analysis

```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \\
     -H "Content-Type: application/json" \\
     -d '{
       "text": "He should be the CEO because men are better leaders.",
       "mode": "fast",
       "cultural_profile": "neutral",
       "include_suggestions": true
     }'
```

### Batch Analysis

```bash
curl -X POST "http://localhost:8000/api/v1/analyze/batch" \\
     -H "Content-Type: application/json" \\
     -d '{
       "texts": [
         "The weather is nice today.",
         "He should be the CEO because men are better leaders.",
         "Old people are not good with technology."
       ],
       "mode": "fast",
       "confidence_threshold": 0.5
     }'
```

### Get Configuration

```bash
curl "http://localhost:8000/api/v1/config"
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
./scripts/test.sh

# Or manually with pytest
pytest tests/ -v --cov=src/bias_engine

# Run specific test categories
pytest tests/unit/ -v           # Unit tests only
pytest tests/integration/ -v   # Integration tests only
```

## 🔧 Configuration

The application can be configured through environment variables or a `.env` file:

### Key Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `false` | Enable debug mode |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `BIAS_THRESHOLD` | `0.7` | Bias detection threshold |
| `CONFIDENCE_THRESHOLD` | `0.8` | Confidence threshold |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |

See `.env.example` for a complete list of configuration options.

## 🏢 Production Deployment

### Docker Production

```bash
# Build production image
docker build -t bias-engine:latest .

# Run with production settings
docker run -d \\
  --name bias-engine \\
  -p 8000:8000 \\
  -e ENVIRONMENT=production \\
  -e DEBUG=false \\
  --restart unless-stopped \\
  bias-engine:latest
```

### Health Checks

The application includes comprehensive health checks for monitoring:

- **Health**: Overall service status
- **Readiness**: Ready to serve traffic
- **Liveness**: Service is alive and responsive

### Monitoring

The application provides structured JSON logging and metrics for monitoring in production environments.

## 🤝 Development

### Code Quality

The project uses several tools for code quality:

```bash
# Code formatting
black src/ tests/

# Import sorting
isort src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📝 API Documentation

When running in development mode (`DEBUG=true`), interactive API documentation is available:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 🔒 Security

- Input validation with Pydantic models
- Request size limits
- Rate limiting (configurable)
- CORS protection
- Trusted host middleware
- Security headers

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:

1. Check the [API documentation](http://localhost:8000/docs)
2. Review the configuration options
3. Check the logs for detailed error information
4. Create an issue with detailed information about the problem