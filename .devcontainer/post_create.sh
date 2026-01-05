#! /usr/bin/env bash
set -e

echo "==> Setting up mlops_exercises devcontainer..."

# Install uv
echo "==> Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Install Dependencies
echo "==> Installing Python dependencies..."
uv sync --dev

# Install pre-commit hooks
echo "==> Installing pre-commit hooks..."
uv run pre-commit install --install-hooks

# DVC setup instructions
echo ""
echo "==> Setup complete!"
echo ""
echo "NOTE: To pull data with DVC, you need to authenticate with GCP:"
echo "  1. Run: gcloud auth application-default login"
echo "  2. Then run: uv run dvc pull"
echo ""
