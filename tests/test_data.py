from pathlib import Path

import torch
from mlops_exercises.data import CorruptMNIST
from torch.utils.data import Dataset

REAL_DATA_PATH = Path("data/processed")


def test_corrupt_mnist_dataset(tmp_path):
    """Test the CorruptMNIST class."""
    # Use real data if available locally, otherwise create mock data
    if (REAL_DATA_PATH / "train_images.pt").exists():
        data_path = REAL_DATA_PATH
        expected_train_len = 30000
        expected_test_len = 5000
    else:
        # Create mock data for CI
        data_path = tmp_path
        expected_train_len = 100
        expected_test_len = 20

        train_images = torch.randn(100, 1, 28, 28)
        train_targets = torch.randint(0, 10, (100,))
        test_images = torch.randn(20, 1, 28, 28)
        test_targets = torch.randint(0, 10, (20,))

        torch.save(train_images, tmp_path / "train_images.pt")
        torch.save(train_targets, tmp_path / "train_target.pt")
        torch.save(test_images, tmp_path / "test_images.pt")
        torch.save(test_targets, tmp_path / "test_target.pt")

    # Test train dataset
    train_dataset = CorruptMNIST(data_path, train=True)
    assert isinstance(train_dataset, Dataset)
    assert len(train_dataset) == expected_train_len

    # Test test dataset
    test_dataset = CorruptMNIST(data_path, train=False)
    assert isinstance(test_dataset, Dataset)
    assert len(test_dataset) == expected_test_len

    # Test __getitem__
    image, target = train_dataset[0]
    assert image.shape == (1, 28, 28)
    assert target.shape == ()
