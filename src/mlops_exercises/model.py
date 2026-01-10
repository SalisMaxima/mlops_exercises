import torch
from torch import nn


class MyAwesomeModel(nn.Module):
    """CNN for corrupted MNIST classification."""

    def __init__(
        self,
        conv_channels: list[int] | None = None,
        fc_hidden: int = 256,
        dropout: float = 0.5,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        if conv_channels is None:
            conv_channels = [32, 64, 128]
        c1, c2, c3 = conv_channels

        # Use padding = kernel_size // 2 to preserve spatial dimensions (for odd kernel sizes)
        padding = kernel_size // 2

        self.conv_layers = nn.Sequential(
            # Input: [batch, 1, 28, 28]
            nn.Conv2d(1, c1, kernel_size=kernel_size, padding=padding),  # -> [batch, c1, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [batch, c1, 14, 14]
            nn.Conv2d(c1, c2, kernel_size=kernel_size, padding=padding),  # -> [batch, c2, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [batch, c2, 7, 7]
            nn.Conv2d(c2, c3, kernel_size=kernel_size, padding=padding),  # -> [batch, c3, 7, 7]
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # -> [batch, c3*7*7]
            nn.Linear(c3 * 7 * 7, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 10),  # -> [batch, 10] (10 digits)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.fc_layers(self.conv_layers(x))


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test with dummy input
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
