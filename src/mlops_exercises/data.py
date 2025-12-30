from pathlib import Path

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.utils.data import Dataset


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images to mean=0, std=1."""
    return (images - images.mean()) / images.std()


class CorruptMNIST(Dataset):
    """Dataset for corrupt MNIST."""

    def __init__(self, data_path: Path, train: bool = True) -> None:
        """Initialize dataset.

        Args:
            data_path: Path to processed data folder
            train: If True, load training data; else load test data
        """
        self.data_path = Path(data_path)

        if train:
            self.images = torch.load(self.data_path / "train_images.pt")
            self.targets = torch.load(self.data_path / "train_target.pt")
        else:
            self.images = torch.load(self.data_path / "test_images.pt")
            self.targets = torch.load(self.data_path / "test_target.pt")

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return image and target at index."""
        return self.images[index], self.targets[index]


def preprocess(raw_dir: Path, output_folder: Path) -> None:
    """Process raw data and save to output folder.

    Args:
        raw_dir: Path to raw data (contains train_images_0.pt, etc.)
        output_folder: Path to save processed data
    """
    print(f"Preprocessing data from {raw_dir}...")

    # Create output folder if it doesn't exist
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load training data (6 files × 5000 samples = 30,000 total)
    train_images, train_target = [], []
    for i in range(6):
        images_path = Path(raw_dir) / f"train_images_{i}.pt"
        target_path = Path(raw_dir) / f"train_target_{i}.pt"

        train_images.append(torch.load(images_path))
        train_target.append(torch.load(target_path))
        print(f"  Loaded {images_path.name}")

    # Concatenate into single tensors
    train_images = torch.cat(train_images)  # Shape: [30000, 28, 28]
    train_target = torch.cat(train_target)  # Shape: [30000]

    # Load test data
    test_images = torch.load(Path(raw_dir) / "test_images.pt")
    test_target = torch.load(Path(raw_dir) / "test_target.pt")
    print("  Loaded test data")

    # Add channel dimension: [N, 28, 28] → [N, 1, 28, 28]
    # Neural networks expect: [batch, channels, height, width]
    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()

    # Convert targets to long (required for classification loss functions)
    train_target = train_target.long()
    test_target = test_target.long()

    # Normalize to mean=0, std=1
    train_images = normalize(train_images)
    test_images = normalize(test_images)

    # Save processed tensors
    torch.save(train_images, output_folder / "train_images.pt")
    torch.save(train_target, output_folder / "train_target.pt")
    torch.save(test_images, output_folder / "test_images.pt")
    torch.save(test_target, output_folder / "test_target.pt")

    print(f"Saved processed data to {output_folder}")
    print(f"  Train: {train_images.shape}, Test: {test_images.shape}")


def corrupt_mnist() -> tuple[Dataset, Dataset]:
    """Return train and test datasets for corrupt MNIST.

    Convenience function to load processed data as TensorDatasets.
    Call preprocess() first if data/processed doesn't exist.
    """
    train_set = CorruptMNIST(Path("data/processed"), train=True)
    test_set = CorruptMNIST(Path("data/processed"), train=False)
    return train_set, test_set


def show_image_and_target(images: torch.Tensor, target: torch.Tensor) -> None:
    """Plot images and their labels in a grid."""
    row_col = int(len(images) ** 0.5)
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
    for ax, im, label in zip(grid, images, target):
        ax.imshow(im.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label.item()}")
        ax.axis("off")
    plt.show()


if __name__ == "__main__":
    train_set, test_set = corrupt_mnist()
    print(f"Size of training set: {len(train_set)}")
    print(f"Size of test set: {len(test_set)}")
    print(f"Shape of a training point {(train_set[0][0].shape, train_set[0][1].shape)}")
    print(f"Shape of a test point {(test_set[0][0].shape, test_set[0][1].shape)}")
    show_image_and_target(train_set.tensors[0][:25], train_set.tensors[1][:25])
