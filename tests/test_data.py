from mlops_exercises.data import CorruptMNIST
from torch.utils.data import Dataset


def test_corrupt_mnist_dataset():
    """Test the CorruptMNIST class."""
    dataset = CorruptMNIST("data/processed", train=True)
    assert isinstance(dataset, Dataset)
    assert len(dataset) > 0
