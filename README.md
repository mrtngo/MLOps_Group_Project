# MLOps Group Project - Cryptocurrency Price Prediction

> A fully automated, containerized, CI/CD-driven MLOps pipeline for real-time cryptocurrency price & direction prediction.

[![CI/CD Pipeline](https://github.com/mrtngo/MLOps_Group_Project/actions/workflows/check.yml/badge.svg)](https://github.com/mrtngo/MLOps_Group_Project/actions/workflows/check.yml)  [![Deploy to Production](https://github.com/mrtngo/MLOps_Group_Project/actions/workflows/publish.yml/badge.svg)](https://github.com/mrtngo/MLOps_Group_Project/actions/workflows/publish.yml)  [![Documentation](https://img.shields.io/badge/documentation-available-brightgreen.svg)](https://mrtngo.github.io/MLOps_Group_Project/)  [![License](https://img.shields.io/github/license/mrtngo/MLOps_Group_Project)](https://github.com/mrtngo/MLOps_Group_Project/blob/main/LICENSE.txt)  [![Release](https://img.shields.io/github/v/release/mrtngo/MLOps_Group_Project)](https://github.com/mrtngo/MLOps_Group_Project/releases)

**🌐 Live API Endpoint:** [https://mlops-group-project.onrender.com](https://mlops-group-project.onrender.com) (supports both single & batch prediction)
**📊 W&B Project Workspace:** [View on Weights & Biases](https://wandb.ai/aviv275-ie-university/mlops-project/workspace?nw=nwuseraviv275)

A comprehensive, production-ready MLOps pipeline for cryptocurrency price prediction and direction classification. This project transforms a notebook workflow into a modular, automated system with full CI/CD integration.

## 🚀 Features

- **Dual Models**: Price prediction (Linear Regression) & direction classification (Logistic Regression)
- **Feature Engineering**: RandomForest-based selection (configurable top‑N features)
- **Data Balancing**: SMOTE oversampling for classification
- **Evaluation**: RMSE, ROC AUC, confusion matrices, plus interactive plots
- **Inference**: Real-time FastAPI with single & batch endpoints

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
- [Roadmap](#roadmap)
- [License](#license)
- [Code of Conduct](#code-of-conduct)
- [Acknowledgments](#acknowledgments)
- [Support](#support)

## 🛠️ Installation

### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or Conda

### Quick Installation

**Using uv (recommended):**
```bash
# Clone repo & install
git clone https://github.com/mrtngo/MLOps_Group_Project.git && cd MLOps_Group_Project
uv sync
```

**Using Conda:**
```bash
conda env create -f environment.yml
conda activate mlops_project
```

### Environment Variables
Create a `.env` file:
```bash
WANDB_PROJECT=mlops-project
WANDB_ENTITY=your-wandb-entity
WANDB_API_KEY=your-wandb-api-key
```

## 🚀 Quick Start

1. **Run the pipeline**
   ```bash
   python main.py            # full workflow
   # or
   just run-pipeline
   ```
2. **Start API server**
   ```bash
   docker build -t crypto-api .
   docker run -p 8000:8000 crypto-api
   # or
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
3. **Test single prediction**
   ```bash
   curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "ETHUSDT_price":1800.0,
       "BNBUSDT_price":300.0,
       "XRPUSDT_price":0.5,
       "ADAUSDT_price":0.3,
       "SOLUSDT_price":25.0,
       "BTCUSDT_funding_rate":0.0001,
       "ETHUSDT_funding_rate":0.0001,
       "BNBUSDT_funding_rate":0.0001,
       "XRPUSDT_funding_rate":0.0001,
       "ADAUSDT_funding_rate":0.0001,
       "SOLUSDT_funding_rate":0.0001
   }'
   ```

**Sample response:**
```json
{
  "price_prediction":45000.0,
  "direction_prediction":1,
  "direction_probability":0.75
}
```

## 📁 Project Structure

```
MLOps_Group_Project/
├── app/               # FastAPI application
│   ├── main.py        # API endpoints
│   └── __init__.py
├── src/mlops/         # Core pipeline modules
├── conf/              # Hydra configs
├── tests/             # Unit & integration tests
├── tasks/             # justfile commands
├── docs/              # Documentation site
├── data/              # Raw, processed & inference data
├── models/            # Saved artifacts
├── Dockerfile         # Container config
├── justfile           # Development commands
├── pyproject.toml     # Project metadata
└── environment.yml    # Conda dependencies
```

*Note: `src/mlops/preprocess/` contains preprocessing logic.*

## 🔧 Usage

Run entire pipeline or stages:
```bash
python main.py           # all stages
python main.py main.steps="data_load,validation,features,preprocess,models"
python main.py main.steps="inference"
```

Override configs:
```bash
python main.py model.active=logistic_regression
dataset.start_date=2024-01-01
```

## 🌐 API Documentation

Visit [Swagger UI](http://localhost:8000/docs) for interactive docs.

## ⚙️ Configuration

See `conf/` for Hydra YAMLs. Key sections:

- **data_source**: API endpoints, file paths
- **model**: active model & parameters
- **feature_engineering**: selection method & params

Override via CLI or custom config paths.

## 🧪 Testing

Run tests:
```bash
pytest        # or just test
```  
Coverage report:
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## 🚀 Deployment

**Docker:**
```bash
docker build -t crypto-api .
docker run -d -p 8000:8000 crypto-api
```
**Render:** automatic GitHub deploy

## 🤝 Contributing

1. Fork & create branch
2. Implement feature
3. **Include tests & docs in your PR**
4. Commit (`just commit`) & push
5. Open PR

Follow PEP 8, conventional commits, and ensure all CI checks pass.

## 📈 Roadmap

- Add deep learning models (LSTM, Transformers)
- Expand supported cryptocurrencies
- Enhance alerting & monitoring dashboards
- Integrate AutoML experiments

## 📄 License

MIT © the contributors. See [LICENSE.txt](LICENSE.txt).

## 📜 Code of Conduct

This project follows the Contributor Covenant. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## 🙏 Acknowledgments

- Binance API for data
- Weights & Biases & MLflow for tracking
- FastAPI & Hydra for infrastructure

## 📞 Support

- **Issues:** https://github.com/mrtngo/MLOps_Group_Project/issues
- **Discussions:** https://github.com/mrtngo/MLOps_Group_Project/discussions
- **Docs:** https://mrtngo.github.io/MLOps_Group_Project/

---

⭐ Star this repo if you find it helpful!

