from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from mlops_exercises.data import corrupt_mnist
from mlops_exercises.model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# MNIST digit labels
MNIST_LABELS = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}

app = typer.Typer()


@app.command()
def embeddings(model_checkpoint: str = "models/model.pth", figure_name: str = "embeddings.png") -> None:
    """Visualize MNIST model embeddings using t-SNE.

    Loads a pre-trained model, extracts intermediate representations (features before
    the final classification layer), and visualizes them in 2D space using t-SNE.
    Each digit class will be shown in a different color.
    """
    print(f"Loading model from {model_checkpoint}")
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
    model.eval()

    # Replace the final classification layer with Identity to get embeddings
    # The model has fc_layers as a Sequential, so we replace the last layer (Linear(256, 10))
    # This gives us the 256-dimensional embeddings before final digit classification
    model.fc_layers[-1] = torch.nn.Identity()

    print("Loading test data")
    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    print("Extracting embeddings")
    embeddings, targets = [], []
    with torch.inference_mode():
        for images, target in test_dataloader:
            images = images.to(DEVICE)
            # Get embeddings (256-dim features before final classification)
            features = model(images)
            embeddings.append(features.cpu())
            targets.append(target)

        embeddings = torch.cat(embeddings).numpy()
        targets = torch.cat(targets).numpy()

    print(f"Embeddings shape: {embeddings.shape}")
    print("Running t-SNE dimensionality reduction (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings)

    print("Creating visualization")
    plt.figure(figsize=(12, 10))
    scatter_plots = []
    for class_id in range(10):
        mask = targets == class_id
        scatter = plt.scatter(
            embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=MNIST_LABELS[class_id], alpha=0.6, s=20
        )
        scatter_plots.append(scatter)

    plt.legend(title="Digit", loc="best", fontsize=9)
    plt.title("t-SNE Visualization of MNIST Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, alpha=0.3)

    # Save to reports/figures
    figures_path = Path(f"reports/figures/{figure_name}")
    figures_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved to {figures_path}")


@app.command()
def explore(figure_name: str = "class_distribution.png") -> None:
    """Explore MNIST dataset distribution and visualize samples.

    Creates a comprehensive view of the dataset including class balance
    and representative samples from each digit class.
    """
    print("Loading MNIST dataset")
    train_set, test_set = corrupt_mnist()

    # Extract labels from datasets
    train_labels = [label for _, label in train_set]
    test_labels = [label for _, label in test_set]

    # Count samples per class
    train_counts = [train_labels.count(i) for i in range(10)]
    test_counts = [test_labels.count(i) for i in range(10)]
    total_counts = [train_counts[i] + test_counts[i] for i in range(10)]

    print("\nDataset Statistics:")
    print(f"Training samples: {len(train_set)}")
    print(f"Test samples: {len(test_set)}")
    print(f"Total samples: {len(train_set) + len(test_set)}")
    print("\nPer-class distribution:")
    for i in range(10):
        print(
            f"  Digit {MNIST_LABELS[i]:2s}: {total_counts[i]:5d} (train: {train_counts[i]:5d}, test: {test_counts[i]:5d})"
        )

    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.3)

    # Top: Class distribution bar chart
    ax_bar = fig.add_subplot(gs[0])
    x_pos = range(10)
    width = 0.35

    bars1 = ax_bar.bar([x - width / 2 for x in x_pos], train_counts, width, label="Train", alpha=0.8)
    bars2 = ax_bar.bar([x + width / 2 for x in x_pos], test_counts, width, label="Test", alpha=0.8)

    ax_bar.set_xlabel("Digit Class", fontsize=12)
    ax_bar.set_ylabel("Number of Samples", fontsize=12)
    ax_bar.set_title("MNIST Class Distribution", fontsize=14, fontweight="bold")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([MNIST_LABELS[i] for i in range(10)])
    ax_bar.legend()
    ax_bar.grid(axis="y", alpha=0.3)

    # Add count labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2.0, height, f"{int(height)}", ha="center", va="bottom", fontsize=8
            )

    # Bottom: Sample grid (10 classes × 10 samples)
    ax_grid = fig.add_subplot(gs[1])
    ax_grid.axis("off")

    # Create grid of subplots for samples
    grid_gs = gs[1].subgridspec(10, 10, hspace=0.05, wspace=0.05)

    print("\nCreating sample grid")
    for class_id in range(10):
        # Get all samples for this class
        class_samples = [(img, idx) for idx, (img, label) in enumerate(train_set) if label == class_id]

        # Select 10 evenly spaced samples
        step = len(class_samples) // 10
        selected_samples = [class_samples[i * step][0] for i in range(10)]

        for sample_id in range(10):
            ax = fig.add_subplot(grid_gs[class_id, sample_id])
            img = selected_samples[sample_id].squeeze().numpy()
            ax.imshow(img, cmap="gray")
            ax.axis("off")

            # Add class label on the first column
            if sample_id == 0:
                ax.text(
                    -0.1,
                    0.5,
                    f"Digit {MNIST_LABELS[class_id]}",
                    transform=ax.transAxes,
                    fontsize=10,
                    va="center",
                    ha="right",
                    fontweight="bold",
                )

    # Save figure
    figures_path = Path(f"reports/figures/{figure_name}")
    figures_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to {figures_path}")


@app.command()
def confusion(
    model_checkpoint: str = "models/model.pth", figure_name: str = "confusion_matrix.png", n_examples: int = 5
) -> None:
    """Generate confusion matrix with misclassified examples.

    Evaluates the model on the test set and creates a confusion matrix
    showing which classes are confused. Includes visual examples of the
    worst misclassifications for error analysis.
    """
    print(f"Loading model from {model_checkpoint}")
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
    model.eval()

    print("Loading test data")
    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    print("Generating predictions")
    all_predictions = []
    all_targets = []
    all_images = []

    with torch.inference_mode():
        for images, targets in test_dataloader:
            images = images.to(DEVICE)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_images.extend(images.cpu())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Calculate accuracy
    accuracy = (all_predictions == all_targets).mean()
    print(f"\nTest accuracy: {accuracy:.4f}")

    # Find misclassified examples
    misclassified_indices = np.where(all_predictions != all_targets)[0]
    print(f"Total misclassified: {len(misclassified_indices)} / {len(all_targets)}")

    # Create figure with confusion matrix and example misclassifications
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.4)

    # Top: Confusion matrix
    ax_cm = fig.add_subplot(gs[0])
    im = ax_cm.imshow(cm_normalized, interpolation="nearest", cmap="Blues")
    ax_cm.set_title(
        f"Confusion Matrix (Normalized)\nOverall Accuracy: {accuracy:.2%}", fontsize=14, fontweight="bold", pad=20
    )

    cbar = plt.colorbar(im, ax=ax_cm)
    cbar.set_label("Proportion", rotation=270, labelpad=20)

    # Add labels
    tick_marks = np.arange(10)
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_xticklabels([MNIST_LABELS[i] for i in range(10)])
    ax_cm.set_yticklabels([MNIST_LABELS[i] for i in range(10)])
    ax_cm.set_xlabel("Predicted Label", fontsize=12)
    ax_cm.set_ylabel("True Label", fontsize=12)

    # Add text annotations
    thresh = cm_normalized.max() / 2
    for i in range(10):
        for j in range(10):
            ax_cm.text(
                j,
                i,
                f"{cm_normalized[i, j]:.2f}\n({cm[i, j]})",
                ha="center",
                va="center",
                color="white" if cm_normalized[i, j] > thresh else "black",
                fontsize=8,
            )

    # Bottom: Misclassified examples
    ax_examples = fig.add_subplot(gs[1])
    ax_examples.axis("off")
    ax_examples.set_title("Example Misclassifications", fontsize=14, fontweight="bold", pad=10)

    # Create grid for misclassified examples
    n_show = min(n_examples, len(misclassified_indices))
    examples_gs = gs[1].subgridspec(1, n_show, wspace=0.3)

    # Select diverse misclassifications
    if len(misclassified_indices) > n_show:
        selected_indices = np.random.choice(misclassified_indices, n_show, replace=False)
    else:
        selected_indices = misclassified_indices

    for idx, mis_idx in enumerate(selected_indices[:n_show]):
        ax = fig.add_subplot(examples_gs[0, idx])
        img = all_images[mis_idx].squeeze().numpy()
        ax.imshow(img, cmap="gray")
        ax.axis("off")

        true_label = MNIST_LABELS[all_targets[mis_idx]]
        pred_label = MNIST_LABELS[all_predictions[mis_idx]]

        ax.set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=9, color="red")

    # Save figure
    figures_path = Path(f"reports/figures/{figure_name}")
    figures_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to {figures_path}")


@app.command()
def all(model_checkpoint: str = "models/model.pth") -> None:
    """Generate all visualizations at once.

    Runs explore, confusion, and embeddings commands to create
    a complete set of visualization outputs.
    """
    print("Generating all visualizations...\n")

    print("=" * 60)
    print("1/3: Class Distribution")
    print("=" * 60)
    explore()

    print("\n" + "=" * 60)
    print("2/3: Confusion Matrix")
    print("=" * 60)
    confusion(model_checkpoint)

    print("\n" + "=" * 60)
    print("3/3: t-SNE Embeddings")
    print("=" * 60)
    embeddings(model_checkpoint)

    print("\n" + "=" * 60)
    print("All visualizations complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - reports/figures/class_distribution.png")
    print("  - reports/figures/confusion_matrix.png")
    print("  - reports/figures/embeddings.png")


if __name__ == "__main__":
    app()
