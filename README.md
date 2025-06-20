# MLOps Group Project

**Live API Endpoint:** [https://mlops-group-project.onrender.com](https://mlops-group-project.onrender.com)

[![check.yml](https://github.com/mrtngo/MLOps_Group_Project/actions/workflows/check.yml/badge.svg)](https://github.com/mrtngo/MLOps_Group_Project/actions/workflows/check.yml)
[![publish.yml](https://github.com/mrtngo/MLOps_Group_Project/actions/workflows/publish.yml/badge.svg)](https://github.com/mrtngo/MLOps_Group_Project/actions/workflows/publish.yml)
[![Documentation](https://img.shields.io/badge/documentation-available-brightgreen.svg)](https://mrtngo.github.io/MLOps_Group_Project/)
[![License](https://img.shields.io/github/license/mrtngo/MLOps_Group_Project)](https://github.com/mrtngo/MLOps_Group_Project/blob/main/LICENCE.txt)
[![Release](https://img.shields.io/github/v/release/mrtngo/MLOps_Group_Project)](https://github.com/mrtngo/MLOps_Group_Project/releases)

A comprehensive MLOps pipeline for cryptocurrency price prediction and direction classification. This project transforms Jupyter notebook workflows into a production-ready, modularized machine learning system.

## Overview

This project implements an end-to-end machine learning pipeline that:

- Fetches cryptocurrency data from Binance API (spot prices and funding rates)
- Validates and preprocesses the data with configurable schemas
- Trains both regression (price prediction) and classification (direction prediction) models
- Evaluates model performance with comprehensive metrics and visualizations
- Provides inference capabilities for new data

## Features

- **Data Loading**: Automated data fetching from Binance spot and futures APIs
- **Data Validation**: Schema-based validation with configurable error handling
- **Feature Engineering**: Automated feature selection using RandomForest importance
- **Model Training**: Linear regression for price prediction and logistic regression for direction classification
- **Preprocessing Pipeline**: Standardization, SMOTE oversampling, and feature selection
- **Model Evaluation**: Comprehensive metrics including RMSE, ROC AUC, confusion matrices, and visualizations
- **Inference Engine**: Production-ready inference with preprocessing pipeline preservation
- **Configuration Management**: YAML-based configuration for all pipeline parameters

## Installation

Use the package manager [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Alternatively, you can install dependencies using conda:

```bash
conda env create -f environment.yml
conda activate mlops_project
```

## Usage

### Full Pipeline Training

Run the complete training pipeline:

**Unix/Linux/macOS:**
```bash
PYTHONPATH=src python3 src/mlops/main.py
```

**Windows (PowerShell):**
```cmd
cmd /c "set PYTHONPATH=src && python src/mlops/main.py"
```

### Inference

Run inference on new data:

**Unix/Linux/macOS:**
```bash
PYTHONPATH=src python3 src/mlops/main.py --stage infer --start-date 2024-01-01 --end-date 2024-01-31 --output-csv predictions.csv
```

**Windows (PowerShell):**
```cmd
cmd /c "set PYTHONPATH=src && python src/mlops/main.py --stage infer --start-date 2024-01-01 --end-date 2024-01-31 --output-csv predictions.csv"
```

### Command Line Options

```bash
python src/mlops/main.py [OPTIONS]

Options:
  --stage {all,infer}           Pipeline stage to run (default: all)
  --output-csv PATH             Output CSV file for inference stage
  --config PATH                 Path to YAML configuration file (default: config.yaml)
  --start-date YYYY-MM-DD       Start date for data fetching (default: 2023-01-01)
  --end-date YYYY-MM-DD         End date for data fetching (default: 2023-12-31)
```

## Pipeline Components

### Data Loading
- Fetches cryptocurrency klines (OHLCV) data from Binance spot API
- Retrieves funding rates from Binance futures API
- Supports configurable date ranges and symbols
- Implements rate limiting and error handling

### Data Validation
- Schema-based validation with type checking
- Range validation for numerical features
- Missing value detection and handling strategies
- Configurable error handling (warn/raise)

### Feature Engineering
- Automatic feature and target definition from configuration
- Price direction labeling for classification tasks
- RandomForest-based feature selection
- Configurable feature selection parameters

### Model Training
- Linear regression for continuous price prediction
- Logistic regression for binary direction classification
- Automated preprocessing pipeline with StandardScaler
- SMOTE oversampling for imbalanced classification data
- Model persistence and artifact management

### Evaluation
- Regression metrics: RMSE
- Classification metrics: Accuracy, F1 Score, ROC AUC
- Confusion matrix visualization
- Price prediction time series plots
- JSON metrics reporting

### Inference
- Production-ready inference engine
- Preprocessing pipeline preservation and application
- Batch prediction capabilities
- Multiple output formats

## Configuration

The pipeline is configured through `config.yaml`. Key sections include:

- **data_source**: API endpoints and data paths
- **symbols**: Cryptocurrency pairs to analyze
- **data_validation**: Schema definitions and validation rules
- **preprocessing**: Scaling and sampling parameters
- **feature_engineering**: Feature selection configuration
- **model**: Model parameters and save paths
- **logging**: Logging configuration

## Core Project Structure

```
src/mlops/
├── data_load/          # Data fetching and loading
├── data_validation/    # Data validation and schema checking
├── features/          # Feature engineering and selection
├── preproccess/       # Data preprocessing (scaling, SMOTE)
├── models/           # Model training and management
├── evaluation/       # Model evaluation and metrics
├── inference/        # Production inference engine
└── main.py          # Main pipeline orchestrator

tests/                # Comprehensive test suite
config.yaml          # Configuration file
```

## Supported Cryptocurrencies

By default, the pipeline supports:
- BTCUSDT (target for prediction)
- ETHUSDT
- BNBUSDT
- XRPUSDT
- ADAUSDT
- SOLUSDT

Additional symbols can be configured in `config.yaml`.

## Output Artifacts

The pipeline generates the following artifacts with their default storage locations:

### Models and Preprocessing
- **Trained models**: `models/linear_regression.pkl`, `models/logistic_regression.pkl`
- **Preprocessing pipeline**: `models/preprocessing_pipeline.pkl`
- **Feature selections and scaler**: Stored within preprocessing pipeline

### Data and Splits
- **Processed data**: `./data/processed/futures_data_processed_.csv`
- **Data splits**: `data/splits/` (when configured)

### Evaluation and Reporting
- **Evaluation metrics**: `models/metrics.json` (or `reports/evaluation_metrics.json`)
- **Validation reports**: `logs/validation_report.json`
- **Confusion matrix plot**: `plots/confusion_matrix.png`
- **Price prediction visualization**: `plots/price_prediction_plot.png`

### Logs and Outputs
- **Application logs**: `./logs/main.log`
- **Inference predictions**: Specified by `--output-csv` parameter (default: `data/processed/output.csv`)

All output paths are configurable through `config.yaml` in the respective sections (`artifacts`, `data_source`, `logging`, etc.).

## API and Docker Usage

### Running with Docker

To build and run the application as a Docker container, use the following commands:

```bash
# 1. Build the Docker image
docker build -t crypto-prediction-api .

# 2. Run the Docker container
docker run -d -p 8000:8000 crypto-prediction-api
```

The API will be accessible at `http://localhost:8000`.

### Testing the API

You can test the running API using the `scripts/call_api.py` script:
```bash
python scripts/call_api.py
```

Alternatively, you can use `curl` to send a request directly to the `/predict` endpoint:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
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

## Testing

Run the test suite:
