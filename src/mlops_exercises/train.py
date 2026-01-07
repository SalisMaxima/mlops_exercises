import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
from loguru import logger
from omegaconf import DictConfig

from mlops_exercises.data import corrupt_mnist
from mlops_exercises.model import MyAwesomeModel


def configure_logging(output_dir: str) -> None:
    """Configure loguru to save logs in Hydra's output directory."""
    logger.remove()  # Remove default handler

    # Add file handler that saves to Hydra's output directory
    log_path = Path(output_dir) / "training.log"
    logger.add(
        log_path,
        level="DEBUG",
        rotation="100 MB",
        retention=5,
        compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )

    # Add console handler for INFO and above
    logger.add(sys.stderr, level="INFO")

    logger.info(f"Logging configured. Logs will be saved to {log_path}")


def get_device() -> torch.device:
    """Get the best available device."""
    # Device priority: CUDA > MPS > CPU
    # Check for CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Check for MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    # Default to CPU if no other device is available
    return torch.device("cpu")


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    """Train a model on MNIST."""
    # Get Hydra's output directory and configure loguru
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    configure_logging(output_dir)

    device = get_device()
    logger.info(f"Training on {device}")
    logger.info(f"Configuration:\n{cfg}")

    # Set seed for reproducibility
    torch.manual_seed(cfg.training.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(cfg.training.seed)

    # Initialize model with config parameters
    model = MyAwesomeModel(
        conv_channels=list(cfg.model.conv_channels),
        fc_hidden=cfg.model.fc_hidden,
        dropout=cfg.model.dropout,
        kernel_size=cfg.model.kernel_size,
    ).to(device)
    logger.info(f"Model:\n{model}")

    # Load data
    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.training.batch_size, shuffle=True)

    # Loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    logger.info(f"Using optimizer: {optimizer.__class__.__name__}")

    statistics = {"train_loss": [], "train_accuracy": []}

    # Training loop
    for epoch in range(cfg.training.epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(device), target.to(device)

            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()

            statistics["train_loss"].append(loss.item())
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                logger.info(f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}, acc: {accuracy:.4f}")

    logger.info("Training complete")

    # Save model (use Hydra output dir)
    model_path = Path(output_dir) / "model.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    # Save training plots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")

    figures_path = Path(output_dir) / "training_statistics.png"
    fig.savefig(figures_path)
    logger.info(f"Training plot saved to {figures_path}")


if __name__ == "__main__":
    train()
