"""
Sweep wrapper script that runs training followed by evaluation.

This script is designed to be used with W&B sweeps. It:
1. Initializes a single wandb run for the sweep
2. Runs training with the sweep hyperparameters
3. Evaluates on the test set
4. Logs test_accuracy for sweep optimization

Usage:
    wandb sweep configs/sweep.yaml --project mlops_exercises
    wandb agent <sweep-id>
"""

import sys
from pathlib import Path

import hydra
import torch
import wandb
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from loguru import logger
from omegaconf import OmegaConf

from mlops_exercises.evaluate import evaluate_model
from mlops_exercises.train import configure_logging, get_device, train_model

# Find project root and config path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = str(_PROJECT_ROOT / "configs")


def main():
    """Main entry point for sweep runs."""
    # Parse command line args for Hydra overrides (from sweep)
    # Args come in as key=value pairs from wandb sweep
    overrides = sys.argv[1:]

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    # Initialize Hydra with compose API
    initialize_config_dir(config_dir=_CONFIG_PATH, version_base="1.3")
    cfg = compose(config_name="config", overrides=overrides)

    # Create output directory for this run
    output_dir = Path(_PROJECT_ROOT) / "outputs" / "sweeps" / wandb.util.generate_id()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    configure_logging(str(output_dir))

    device = get_device()

    # Initialize wandb for the sweep (single run for train + eval)
    run = wandb.init(
        project="mlops_exercises",
        job_type="sweep",
        reinit=True,
        config={
            "learning_rate": cfg.optimizer.lr,
            "batch_size": cfg.training.batch_size,
            "epochs": cfg.training.epochs,
            "seed": cfg.training.seed,
            "conv_channels": list(cfg.model.conv_channels),
            "fc_hidden": cfg.model.fc_hidden,
            "dropout": cfg.model.dropout,
            "kernel_size": cfg.model.kernel_size,
            "architecture": "MyAwesomeModel",
            "dataset": "corrupt_mnist",
        },
    )

    logger.info(f"Starting sweep run: {run.id}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Phase 1: Training
    logger.info("=" * 50)
    logger.info("PHASE 1: TRAINING")
    logger.info("=" * 50)

    model_path = train_model(
        cfg=cfg,
        output_dir=str(output_dir),
        device=device,
        wandb_run=run,
    )

    # Phase 2: Evaluation
    logger.info("=" * 50)
    logger.info("PHASE 2: EVALUATION")
    logger.info("=" * 50)

    # Build model config for evaluation
    model_config = {
        "conv_channels": list(cfg.model.conv_channels),
        "fc_hidden": cfg.model.fc_hidden,
        "dropout": cfg.model.dropout,
        "kernel_size": cfg.model.kernel_size,
    }

    test_accuracy = evaluate_model(
        model_checkpoint=model_path,
        model_config=model_config,
        device=device,
        log_to_wandb=True,
    )

    logger.info(f"Sweep run complete. Test accuracy: {test_accuracy:.4f}")

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
