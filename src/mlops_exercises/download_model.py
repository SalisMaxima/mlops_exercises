"""
Download a model from the W&B Model Registry.

Simple script to download a model artifact by alias from the registry.

Usage:
    python download_model.py --alias best --collection "Best-performing-models"
    python download_model.py --alias top-1 --verify
"""

from pathlib import Path

import torch
import typer
import wandb
from loguru import logger

from mlops_exercises.data import corrupt_mnist
from mlops_exercises.model import MyAwesomeModel

app = typer.Typer()


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@app.command()
def download_model(
    alias: str = typer.Option("best", "--alias", "-a", help="Model alias (e.g., 'best', 'top-1', 'latest')"),
    collection: str = typer.Option("Best-performing-models", "--collection", "-c", help="Registry collection name"),
    output: str = typer.Option("models/registry", "--output", "-o", help="Output directory"),
    verify: bool = typer.Option(False, "--verify", "-v", help="Run inference to verify the model"),
) -> None:
    """Download a model from the W&B Model Registry."""
    api = wandb.Api()
    entity = api.default_entity

    # Build artifact path
    artifact_path = f"{entity}/model-registry/{collection}:{alias}"
    logger.info(f"Downloading artifact: {artifact_path}")

    try:
        artifact = api.artifact(artifact_path)
    except wandb.errors.CommError as e:
        logger.error(f"Failed to find artifact: {e}")
        raise typer.Exit(1) from None

    # Download artifact
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = artifact.download(str(output_dir))
    logger.info(f"Downloaded to: {artifact_dir}")

    # Log metadata
    if artifact.metadata:
        logger.info(f"Artifact metadata: {artifact.metadata}")

    # Find model file
    model_file = None
    for file in Path(artifact_dir).iterdir():
        if file.suffix in [".pth", ".pt", ".ckpt"]:
            model_file = file
            break

    if model_file is None:
        logger.error(f"No model file found in {artifact_dir}")
        raise typer.Exit(1)

    logger.info(f"Model file: {model_file}")

    # Optionally verify the model
    if verify:
        logger.info("Verifying model with test inference...")
        device = get_device()

        # Get model config from metadata if available
        metadata = artifact.metadata or {}
        model = MyAwesomeModel(
            conv_channels=metadata.get("conv_channels", [32, 64, 128]),
            fc_hidden=metadata.get("fc_hidden", 256),
            dropout=metadata.get("dropout", 0.5),
            kernel_size=metadata.get("kernel_size", 3),
        ).to(device)

        state_dict = torch.load(model_file, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        # Run evaluation on test set
        _, test_set = corrupt_mnist()
        test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

        correct, total = 0, 0
        with torch.no_grad():
            for img, target in test_dataloader:
                img, target = img.to(device), target.to(device)
                y_pred = model(img)
                correct += (y_pred.argmax(dim=1) == target).float().sum().item()
                total += target.size(0)

        accuracy = correct / total
        logger.info(f"Test accuracy: {accuracy:.4f} ({int(correct)}/{int(total)})")

    logger.info("Download complete!")


if __name__ == "__main__":
    app()
