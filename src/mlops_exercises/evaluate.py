from typing import Optional

import torch
import typer
import wandb

from mlops_exercises.data import corrupt_mnist
from mlops_exercises.model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

app = typer.Typer()


def evaluate_model(
    model_checkpoint: str,
    model_config: Optional[dict] = None,
    device: Optional[torch.device] = None,
    log_to_wandb: bool = False,
) -> float:
    """
    Core evaluation logic. Can be called standalone or from a sweep wrapper.

    Args:
        model_checkpoint: Path to the model checkpoint file
        model_config: Optional model configuration dict (conv_channels, fc_hidden, etc.)
                      If None, uses default model architecture
        device: Torch device to evaluate on. If None, uses best available.
        log_to_wandb: Whether to log test_accuracy to wandb (assumes wandb is initialized)

    Returns:
        Test accuracy as a float between 0 and 1
    """
    if device is None:
        device = DEVICE

    print(f"Evaluating model from {model_checkpoint}")

    # Initialize model with config if provided, otherwise use defaults
    if model_config:
        model = MyAwesomeModel(
            conv_channels=model_config.get("conv_channels", [32, 64, 128]),
            fc_hidden=model_config.get("fc_hidden", 256),
            dropout=model_config.get("dropout", 0.5),
            kernel_size=model_config.get("kernel_size", 3),
        ).to(device)
    else:
        model = MyAwesomeModel().to(device)

    model.load_state_dict(torch.load(model_checkpoint, map_location=device, weights_only=True))

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for img, target in test_dataloader:
            img, target = img.to(device), target.to(device)
            y_pred = model(img)
            correct += (y_pred.argmax(dim=1) == target).float().sum().item()
            total += target.size(0)

    test_accuracy = correct / total
    print(f"Test accuracy: {test_accuracy:.4f} ({int(correct)}/{int(total)})")

    # Log to wandb if requested (assumes wandb.init() was already called)
    if log_to_wandb:
        wandb.log({"test_accuracy": test_accuracy})
        # Also set as summary metric for sweep optimization
        wandb.run.summary["test_accuracy"] = test_accuracy

    return test_accuracy


@app.command()
def evaluate(
    model_checkpoint: str = typer.Argument("models/model.pth"),
    use_wandb: bool = typer.Option(False, "--wandb", help="Log results to W&B"),
) -> None:
    """Evaluate a trained model (standalone entry point)."""
    if use_wandb:
        wandb.init(
            project="mlops_exercises",
            job_type="eval",
            reinit=True,
        )

    test_accuracy = evaluate_model(
        model_checkpoint=model_checkpoint,
        log_to_wandb=use_wandb,
    )

    if use_wandb:
        wandb.finish()

    return test_accuracy


if __name__ == "__main__":
    app()
