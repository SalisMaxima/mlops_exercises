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

# Install gcloud SDK for DVC GCS support
echo "==> Installing Google Cloud SDK..."
curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=$HOME
echo 'source $HOME/google-cloud-sdk/path.bash.inc' >> $HOME/.bashrc

# DVC setup instructions
echo ""
echo "==> Setup complete!"
echo ""
echo "NOTE: To pull data with DVC, you need to authenticate with GCP:"
echo "  1. Open a new terminal (or run: source ~/.bashrc)"
echo "  2. Run: gcloud auth application-default login"
echo "  3. Then run: uv run dvc pull"
echo ""
