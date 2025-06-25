#!/bin/bash

set -e

echo "🚀 Starting Crypto MLOps Development Environment"

# Create necessary directories
mkdir -p data/processed data/raw models logs plots conf mlruns wandb

# Check if conda environment exists
if ! conda env list | grep -q "mlops_project"; then
    echo "📦 Creating conda environment..."
    conda env create -f environment.yml
else
    echo "✅ Conda environment already exists"
fi

# Activate environment
echo "🔧 Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate mlops_project

# Set environment variables
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
echo "📝 Set PYTHONPATH to include src/"

echo ""
echo "🎉 Development environment is ready!"
echo ""
echo "🔧 Useful commands:"
echo "  conda activate mlops_project    # Activate environment"
echo "  python main.py                  # Run full pipeline"
echo "  python main.py main.steps=model # Run specific step"
echo ""