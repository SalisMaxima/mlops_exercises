#! /usr/bin/env bash
set -e

echo "==> Setting up mlops_exercises devcontainer..."

# Install uv
echo "==> Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Check for GPU availability
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "==> NVIDIA GPU detected, installing CUDA PyTorch..."
    uv sync --dev
else
    echo "==> No GPU detected, installing CPU-only PyTorch..."
    # Override torch sources to use CPU-only index
    UV_INDEX_PYTORCH_CU130_URL="https://download.pytorch.org/whl/cpu" uv sync --dev
fi

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
