import torch
from torch import nn


class MyAwesomeModel(nn.Module):
    """CNN for corrupted MNIST classification."""

    def __init__(self) -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(
            # Input: [batch, 1, 28, 28]
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # -> [batch, 32, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [batch, 32, 14, 14]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> [batch, 64, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [batch, 64, 7, 7]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # -> [batch, 128, 7, 7]
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # -> [batch, 128*7*7]
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),  # -> [batch, 10] (10 digits)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


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
