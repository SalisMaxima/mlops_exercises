"""Download and load a model from W&B Model Registry."""

import shutil
import sys
from pathlib import Path

import hydra
import torch
import wandb
from loguru import logger
from omegaconf import DictConfig

from mlops_exercises.evaluate import evaluate_model
from mlops_exercises.model import MyAwesomeModel

# Compute absolute path to configs directory (project_root/configs)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = str(_PROJECT_ROOT / "configs")


def configure_logging() -> None:
    """Configure loguru for console output."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_artifact_name(wandb_cfg: DictConfig) -> tuple[str, str]:
    """Build the artifact name from config.

    Args:
        wandb_cfg: The wandb configuration section.

    Returns:
        Tuple of (artifact_name, display_name for saving).
    """
    # Option 1: Full artifact path provided directly
    artifact_path = wandb_cfg.get("artifact_path")
    if artifact_path:
        # Extract model name from path for saving
        # Format: entity/wandb-registry-collection/model_name:alias
        parts = artifact_path.split("/")
        model_part = parts[-1].split(":")[0] if len(parts) >= 3 else "model"
        return artifact_path, model_part

    # Option 2: Build from components
    entity = wandb_cfg.get("entity")
    collection = wandb_cfg.get("collection")
    registry_name = wandb_cfg.registry_name
    alias = wandb_cfg.alias

    # Get entity from API if not provided
    if not entity:
        api = wandb.Api()
        entity = api.default_entity

    # Build artifact path based on whether collection is specified
    if collection:
        # W&B Model Registry format: entity/wandb-registry-collection/model_name:alias
        artifact_name = f"{entity}/wandb-registry-{collection}/{registry_name}:{alias}"
    else:
        # Try standard model-registry format first
        artifact_name = f"{entity}/model-registry/{registry_name}:{alias}"

    return artifact_name, registry_name


@hydra.main(config_path=_CONFIG_PATH, config_name="consume", version_base="1.3")
def consume_model(cfg: DictConfig) -> MyAwesomeModel:
    """Download and load a model from W&B Model Registry.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Loaded PyTorch model ready for inference.
    """
    configure_logging()

    logger.info(f"Configuration:\n{cfg}")

    # Extract config values
    wandb_cfg = cfg.wandb
    model_cfg = cfg.model
    output_cfg = cfg.output

    # Build artifact name
    artifact_name, display_name = build_artifact_name(wandb_cfg)
    logger.info(f"Fetching artifact: {artifact_name}")

    # Initialize wandb API and download artifact
    api = wandb.Api()
    try:
        artifact = api.artifact(name=artifact_name)
    except wandb.errors.CommError:
        # Try as a regular artifact (not from model registry)
        entity = wandb_cfg.get("entity") or api.default_entity
        project = wandb_cfg.get("project", "mlops_exercises")
        registry_name = wandb_cfg.registry_name
        alias = wandb_cfg.alias
        artifact_name = f"{entity}/{project}/{registry_name}:{alias}"
        logger.info(f"Model registry not found, trying regular artifact: {artifact_name}")
        artifact = api.artifact(name=artifact_name)

    # Download artifact to local directory
    artifact_dir = Path(output_cfg.get("artifact_dir", "artifacts"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact.download(str(artifact_dir))
    logger.info(f"Artifact downloaded to: {artifact_path}")

    # Log artifact metadata
    if artifact.metadata:
        logger.info(f"Artifact metadata: {artifact.metadata}")

    # Find the model file in the downloaded artifact
    model_file = None
    for file in Path(artifact_path).iterdir():
        if file.suffix in [".pth", ".pt", ".ckpt"]:
            model_file = file
            break

    if model_file is None:
        raise FileNotFoundError(f"No model file found in artifact directory: {artifact_path}")

    logger.info(f"Loading model from: {model_file}")

    # Save model to models/ directory
    models_dir = Path(output_cfg.get("model_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    # Use custom model_name if provided, otherwise use display_name from artifact
    save_name = output_cfg.get("model_name") or display_name
    # Sanitize filename (replace spaces with underscores)
    save_name = save_name.replace(" ", "_")
    if not save_name.endswith(model_file.suffix):
        save_name = f"{save_name}{model_file.suffix}"

    saved_model_path = models_dir / save_name
    shutil.copy(model_file, saved_model_path)
    logger.info(f"Model saved to: {saved_model_path}")

    # Build model config dict for loading
    model_config = {
        "conv_channels": list(model_cfg.conv_channels),
        "fc_hidden": model_cfg.fc_hidden,
        "dropout": model_cfg.dropout,
        "kernel_size": model_cfg.kernel_size,
    }

    # Initialize model with config parameters
    device = get_device()
    model = MyAwesomeModel(**model_config).to(device)

    # Load state dict
    state_dict = torch.load(model_file, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(f"Model loaded successfully on {device}")
    logger.info(f"Model architecture:\n{model}")

    # Evaluate on test set if requested (using shared evaluate_model function)
    if output_cfg.get("evaluate", False):
        logger.info("Evaluating model on test set...")
        batch_size = output_cfg.get("batch_size", 32)
        evaluate_model(
            model_checkpoint=str(saved_model_path),
            model_config=model_config,
            device=device,
            batch_size=batch_size,
        )

    return model


if __name__ == "__main__":
    consume_model()
