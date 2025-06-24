# MLOps Group Project - Cryptocurrency Price Prediction

[![CI/CD Pipeline](https://github.com/mrtngo/MLOps_Group_Project/actions/workflows/check.yml/badge.svg)](https://github.com/mrtngo/MLOps_Group_Project/actions/workflows/check.yml)
[![Deploy to Production](https://github.com/mrtngo/MLOps_Group_Project/actions/workflows/publish.yml/badge.svg)](https://github.com/mrtngo/MLOps_Group_Project/actions/workflows/publish.yml)
[![Documentation](https://img.shields.io/badge/documentation-available-brightgreen.svg)](https://mrtngo.github.io/MLOps_Group_Project/)
[![License](https://img.shields.io/github/license/mrtngo/MLOps_Group_Project)](https://github.com/mrtngo/MLOps_Group_Project/blob/main/LICENSE.txt)
[![Release](https://img.shields.io/github/v/release/mrtngo/MLOps_Group_Project)](https://github.com/mrtngo/MLOps_Group_Project/releases)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

**🌐 Live API Endpoint:** [https://mlops-group-project.onrender.com](https://mlops-group-project.onrender.com)  
**📊 W&B Project Workspace:** [View on Weights & Biases](https://wandb.ai/aviv275-ie-university/mlops-project/workspace?nw=nwuseraviv275)

A comprehensive, production-ready MLOps pipeline for cryptocurrency price prediction and direction classification. This project demonstrates the transformation from a Jupyter notebook workflow into a modularized, automated machine learning system with full CI/CD integration.

## 🚀 Features

### Core ML Pipeline
- **📈 Dual Prediction Models**: Linear regression for price prediction and logistic regression for direction classification
- **🔍 Automated Feature Engineering**: RandomForest-based feature selection with configurable parameters
- **⚖️ Data Balancing**: SMOTE oversampling for imbalanced classification datasets
- **📊 Comprehensive Evaluation**: RMSE, ROC AUC, confusion matrices, and interactive visualizations
- **🔮 Production Inference**: Real-time prediction API with batch processing capabilities

### Data Management
- **🌐 Multi-Source Data Ingestion**: Automated fetching from Binance spot and futures APIs
- **✅ Schema Validation**: Configurable data validation with type checking and range validation
- **🔄 Rate Limiting**: Built-in API rate limiting and error handling
- **📋 Data Lineage**: Full traceability from raw data to predictions

### MLOps Infrastructure
- **🔄 CI/CD Pipeline**: Automated testing, formatting, and deployment with GitHub Actions
- **🐳 Containerization**: Docker support for consistent deployment environments
- **📦 MLflow Integration**: Experiment tracking and model versioning
- **📈 W&B Monitoring**: Real-time experiment tracking and model performance monitoring
- **🔧 Hydra Configuration**: YAML-based configuration management with override support

### Development Tools
- **🎯 Code Quality**: Automated formatting (Black, isort), linting (Ruff, Flake8), and type checking (mypy)
- **🧪 Testing**: Comprehensive test suite with pytest and coverage reporting
- **🔒 Security**: Automated security scanning with Bandit
- **📚 Documentation**: Auto-generated API documentation with pdoc
- **⚡ Just Commands**: Simplified development workflow with justfile

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or conda

### Quick Installation

**Using uv (recommended):**
```bash
# Clone the repository
git clone https://github.com/mrtngo/MLOps_Group_Project.git
cd MLOps_Group_Project

# Install dependencies
uv sync
```

**Using conda:**
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate mlops_project
```

### Environment Variables
Create a `.env` file in the project root:
```bash
WANDB_PROJECT=mlops-project
WANDB_ENTITY=your-wandb-entity
WANDB_API_KEY=your-wandb-api-key
```

## 🚀 Quick Start

### 1. Run the Complete Pipeline
```bash
# Using the main orchestrator
python main.py

# Or using just commands
just run-pipeline
```

### 2. Start the API Server
```bash
# Using Docker
docker build -t crypto-prediction-api .
docker run -p 8000:8000 crypto-prediction-api

# Or directly with uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. Make Predictions
```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "ETHUSDT_price": 1800.0,
    "BNBUSDT_price": 300.0,
    "XRPUSDT_price": 0.5,
    "ADAUSDT_price": 0.3,
    "SOLUSDT_price": 25.0,
    "BTCUSDT_funding_rate": 0.0001,
    "ETHUSDT_funding_rate": 0.0001,
    "BNBUSDT_funding_rate": 0.0001,
    "XRPUSDT_funding_rate": 0.0001,
    "ADAUSDT_funding_rate": 0.0001,
    "SOLUSDT_funding_rate": 0.0001
  }'
```

## 📁 Project Structure

```
MLOps_Group_Project/
├── 📁 app/                          # FastAPI application
│   ├── main.py                     # API endpoints and server
│   └── __init__.py
├── 📁 src/                         # Core ML pipeline
│   └── 📁 mlops/
│       ├── 📁 data_load/           # Data fetching and loading
│       ├── 📁 data_validation/     # Schema validation
│       ├── 📁 features/            # Feature engineering
│       ├── 📁 preproccess/         # Data preprocessing
│       ├── 📁 models/              # Model training
│       ├── 📁 evaluation/          # Model evaluation
│       ├── 📁 inference/           # Production inference
│       └── main.py                 # Pipeline orchestrator
├── 📁 conf/                        # Hydra configuration
│   ├── config.yaml                 # Main configuration
│   └── 📁 dataset/                 # Dataset-specific configs
├── 📁 tests/                       # Test suite
├── 📁 scripts/                     # Utility scripts
├── 📁 tasks/                       # Just command definitions
├── 📁 docs/                        # Documentation
├── 📁 models/                      # Trained models and artifacts
├── 📁 data/                        # Data storage
│   ├── 📁 raw/                     # Raw data
│   ├── 📁 processed/               # Processed data
│   └── 📁 inference/               # Inference data
├── 📁 plots/                       # Generated visualizations
├── 📁 reports/                     # Evaluation reports
├── 📁 logs/                        # Application logs
├── 📁 wandb/                       # W&B run artifacts
├── main.py                         # MLflow orchestrator
├── MLproject                       # MLflow project definition
├── Dockerfile                      # Container configuration
├── pyproject.toml                  # Project metadata and tools
├── justfile                        # Development commands
└── requirements.txt                # Python dependencies
```

## 🔧 Usage

### Pipeline Execution

**Full Pipeline:**
```bash
python main.py
```

**Specific Stages:**
```bash
# Training only
python main.py main.steps="data_load,data_validation,features,preprocess,models"

# Inference only
python main.py main.steps="inference"
```

**With Custom Configuration:**
```bash
python main.py data_source.start_date=2024-01-01 data_source.end_date=2024-12-31
```

### Command Line Interface

```bash
python main.py [OPTIONS]

Options:
  --stage {all,infer}           Pipeline stage to run (default: all)
  --output-csv PATH             Output CSV file for inference stage
  --config PATH                 Path to YAML configuration file
  --start-date YYYY-MM-DD       Start date for data fetching
  --end-date YYYY-MM-DD         End date for data fetching
```

### Just Commands

```bash
# Development workflow
just format                    # Format code with Black and isort
just check-code               # Run linting and type checking
just test                     # Run tests with coverage
just doc                      # Generate documentation

# MLflow operations
just mlflow-run              # Run MLflow project
just mlflow-ui               # Start MLflow UI

# Docker operations
just docker-build            # Build Docker image
just docker-run              # Run Docker container

# Package management
just install                 # Install dependencies
just clean                   # Clean build artifacts
```

## 🌐 API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict_batch` | POST | Batch predictions from CSV |

### Request Format

**Single Prediction:**
```json
{
  "ETHUSDT_price": 1800.0,
  "BNBUSDT_price": 300.0,
  "XRPUSDT_price": 0.5,
  "ADAUSDT_price": 0.3,
  "SOLUSDT_price": 25.0,
  "BTCUSDT_funding_rate": 0.0001,
  "ETHUSDT_funding_rate": 0.0001,
  "BNBUSDT_funding_rate": 0.0001,
  "XRPUSDT_funding_rate": 0.0001,
  "ADAUSDT_funding_rate": 0.0001,
  "SOLUSDT_funding_rate": 0.0001
}
```

**Response:**
```json
{
  "price_prediction": 45000.0,
  "direction_prediction": 1,
  "direction_probability": 0.75
}
```

### Interactive Documentation
Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI).

## ⚙️ Configuration

The pipeline is configured through Hydra-managed YAML files in the `conf/` directory.

### Key Configuration Sections

**Data Source:**
```yaml
data_source:
  raw_path_futures: "https://fapi.binance.com/fapi/v1/fundingRate"
  raw_path_spot: "https://api.binance.com/api/v3/klines"
  processed_path: "./data/processed/futures_data_processed_.csv"
```

**Model Configuration:**
```yaml
model:
  active: linear_regression
  linear_regression:
    save_path: models/linear_regression.pkl
    params:
      fit_intercept: true
  logistic_regression:
    save_path: models/logistic_regression.pkl
    params:
      penalty: "l2"
      solver: "lbfgs"
```

**Feature Engineering:**
```yaml
feature_engineering:
  feature_selection:
    method: random_forest
    params:
      n_estimators: 20
      random_state: 42
    top_n: 8
```

### Configuration Overrides

```bash
# Override specific parameters
python main.py model.active=logistic_regression

# Use different configuration files
python main.py --config-path conf/experiment_config.yaml

# Override multiple parameters
python main.py data_source.start_date=2024-01-01 preprocessing.scaling.method=minmax
```

## 🧪 Testing

### Run All Tests
```bash
# Using pytest directly
pytest

# Using just commands
just test
```

### Test Categories
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# API tests
pytest tests/test_api.py

# With coverage
pytest --cov=src --cov-report=html
```

### Test Coverage
```bash
# Generate coverage report
just check-coverage

# View coverage in browser
open htmlcov/index.html
```

## 🚀 Deployment

### Docker Deployment

**Build and Run:**
```bash
# Build image
docker build -t crypto-prediction-api .

# Run container
docker run -d -p 8000:8000 crypto-prediction-api

# With environment variables
docker run -d -p 8000:8000 \
  -e WANDB_PROJECT=mlops-project \
  -e WANDB_ENTITY=your-entity \
  crypto-prediction-api
```

**Docker Compose:**
```bash
docker-compose up -d
```

### Production Deployment

**Render (Current):**
- Automatic deployment from GitHub
- Environment variables configured in Render dashboard
- Health checks enabled

**Other Platforms:**
```bash
# Heroku
heroku create your-app-name
git push heroku main

# AWS ECS
aws ecs create-service --cluster your-cluster --service-name mlops-service

# Google Cloud Run
gcloud run deploy mlops-api --source .
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `WANDB_PROJECT` | W&B project name | Yes |
| `WANDB_ENTITY` | W&B entity/username | Yes |
| `WANDB_API_KEY` | W&B API key | Yes |
| `MLFLOW_TRACKING_URI` | MLflow tracking server | No |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | No |

## 🔧 Development

### Setup Development Environment

```bash
# Install development dependencies
uv sync --group=check,commit,dev,doc

# Install pre-commit hooks
pre-commit install

# Setup git hooks
just setup-hooks
```

### Code Quality Tools

```bash
# Format code
just format

# Lint code
just check-code

# Type checking
just check-type

# Security scanning
just check-security

# All checks
just check
```

### Git Workflow

```bash
# Conventional commits
just commit

# Create release
just release

# Update changelog
just changelog
```

### Documentation

```bash
# Generate documentation
just doc

# Serve documentation locally
just doc-serve
```

## 📊 Monitoring and Logging

### W&B Integration
- **Experiment Tracking**: Automatic logging of hyperparameters, metrics, and artifacts
- **Model Versioning**: Version control for trained models
- **Performance Monitoring**: Real-time model performance tracking
- **Artifact Management**: Centralized storage of models and data

### MLflow Integration
- **Experiment Management**: Organize and track ML experiments
- **Model Registry**: Version and deploy models
- **Pipeline Orchestration**: Coordinate multi-step ML workflows

### Logging
- **Structured Logging**: JSON-formatted logs with configurable levels
- **File Rotation**: Automatic log rotation to prevent disk space issues
- **Error Tracking**: Comprehensive error logging with stack traces

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`just commit`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Write comprehensive tests for new features
- Update documentation for API changes
- Use conventional commit messages
- Ensure all CI checks pass

### Code Review Process

1. Automated checks must pass (formatting, linting, tests)
2. Code review by maintainers
3. Documentation updates if needed
4. Merge after approval

## 📈 Performance

### Model Performance
- **Linear Regression**: RMSE ~$500-1000 for BTC price prediction
- **Logistic Regression**: ROC AUC ~0.65-0.75 for direction prediction
- **Training Time**: ~30-60 seconds for full pipeline
- **Inference Time**: <100ms per prediction

### System Performance
- **API Response Time**: <200ms average
- **Concurrent Requests**: 100+ requests/second
- **Memory Usage**: ~500MB for full pipeline
- **Disk Usage**: ~1GB for models and data

## 🔒 Security

### Security Measures
- **Input Validation**: Pydantic models for request validation
- **Rate Limiting**: API rate limiting to prevent abuse
- **Error Handling**: Secure error messages without sensitive information
- **Dependency Scanning**: Regular security updates for dependencies

### Security Scanning
```bash
# Run security checks
just check-security

# Update dependencies
just update-deps
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## 🙏 Acknowledgments

- **Binance API** for cryptocurrency data
- **Weights & Biases** for experiment tracking
- **MLflow** for model lifecycle management
- **FastAPI** for the web framework
- **Hydra** for configuration management

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/mrtngo/MLOps_Group_Project/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mrtngo/MLOps_Group_Project/discussions)
- **Documentation**: [Project Documentation](https://mrtngo.github.io/MLOps_Group_Project/)

---

**⭐ Star this repository if you find it helpful!**
