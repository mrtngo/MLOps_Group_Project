#!/bin/bash

set -e

echo "🤖 Starting Crypto Model Training..."

# Parse command line arguments
START_DATE=${1:-"2023-01-01"}
END_DATE=${2:-"2023-12-31"}

echo "📅 Training period: $START_DATE to $END_DATE"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate mlops_project

# Set Python path
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Run training pipeline with Hydra
echo "🚀 Starting crypto training pipeline..."
python main.py \
    data_source.start_date="$START_DATE" \
    data_source.end_date="$END_DATE" \
    main.steps="data_load,data_validation,model,evaluation"

echo "✅ Crypto training completed!"
echo "📊 Check results:"
echo "  - Model metrics: cat models/metrics.json"
echo "  - Validation report: cat logs/validation_report.json"