#!/bin/bash
set -e

echo "========================================"
echo "Pulling data from DVC remote..."
echo "========================================"

# Pull data from DVC remote storage
dvc pull

echo "========================================"
echo "Data pulled successfully!"
echo "Starting training..."
echo "========================================"

# Execute the training script with all passed arguments
exec python -u src/mlops_exercises/train.py "$@"
