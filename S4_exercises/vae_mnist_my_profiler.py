"""Adapted from https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb.

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST with profiling.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.utils import save_image

# Profiler configuration
ENABLE_PROFILER = True
PROFILE_BATCHES = 20

# Model Hyperparameters
dataset_path = "datasets"
device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE = torch.device(device_name)
batch_size = 100
x_dim = 784
hidden_dim = 400
latent_dim = 20
lr = 1e-3
epochs = 5
num_workers = 1

print(f"Using device: {DEVICE}")
print(f"Profiler enabled: {ENABLE_PROFILER}")


# Data loading
train_dataset = MNIST(dataset_path, train=True, download=True)
test_dataset = MNIST(dataset_path, train=False, download=True)

train_dataset = TensorDataset(train_dataset.data.float() / 255.0, train_dataset.targets)
test_dataset = TensorDataset(test_dataset.data.float() / 255.0, test_dataset.targets)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=num_workers,
    persistent_workers=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=num_workers,
    persistent_workers=True,
)


class Encoder(nn.Module):
    """Gaussian MLP Encoder."""

    def __init__(self, input_dim, hidden_dim, latent_dim) -> None:
        super().__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        self.training = True

    def forward(self, x):
        """Forward pass."""
        h_ = torch.relu(self.FC_input(x))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)

        std = torch.exp(0.5 * log_var)
        z = self.reparameterization(mean, std)

        return z, mean, log_var

    def reparameterization(self, mean, std):
        """Reparameterization trick."""
        epsilon = torch.randn_like(std)
        return mean + std * epsilon


class Decoder(nn.Module):
    """Bernoulli MLP Decoder."""

    def __init__(self, latent_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass."""
        h = torch.relu(self.FC_hidden(x))
        return torch.sigmoid(self.FC_output(h))


class Model(nn.Module):
    """VAE Model."""

    def __init__(self, encoder, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        """Forward pass."""
        z, mean, log_var = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat, mean, log_var


encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)

model = Model(encoder=encoder, decoder=decoder).to(DEVICE)


BCE_loss = nn.BCELoss()


def loss_function(x, x_hat, mean, log_var):
    """Reconstruction + KL divergence losses summed over all elements and batch."""
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + kld


optimizer = Adam(model.parameters(), lr=lr)

# Setup profiler
if ENABLE_PROFILER:
    profiler_schedule = schedule(skip_first=5, wait=2, warmup=1, active=6, repeat=1)

print("Start training VAE...")
if ENABLE_PROFILER:
    print("Profiler enabled - results will be saved to: ./log/vae_training/")

model.train()

# Start profiler
if ENABLE_PROFILER:
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        if torch.cuda.is_available()
        else [ProfilerActivity.CPU],
        schedule=profiler_schedule,
        on_trace_ready=tensorboard_trace_handler("./log/vae_training"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )
    prof.__enter__()

# Training loop
total_steps = 0
for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}")

        x = x.view(batch_size, x_dim)
        x = x.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var)
        overall_loss += loss.item()
        loss.backward()
        optimizer.step()

        if ENABLE_PROFILER:
            prof.step()

        total_steps += 1

        if ENABLE_PROFILER and total_steps >= PROFILE_BATCHES:
            prof.__exit__(None, None, None)
            ENABLE_PROFILER = False
            print(f"\nProfiling complete! View results: tensorboard --logdir=./log/vae_training\n")

    print(
        "\tEpoch",
        epoch + 1,
        "complete!",
        "\tAverage Loss: ",
        overall_loss / (batch_idx * batch_size),
    )

print("Finish!!")

if ENABLE_PROFILER:
    prof.__exit__(None, None, None)

# Generate reconstructions
model.eval()
with torch.no_grad():
    for batch_idx, (x, _) in enumerate(test_loader):
        if batch_idx % 100 == 0:
            print(batch_idx)
        x = x.view(batch_size, x_dim)
        x = x.to(DEVICE, non_blocking=True)
        x_hat, _, _ = model(x)
        break

save_image(x.view(batch_size, 1, 28, 28), "orig_data.png")
save_image(x_hat.view(batch_size, 1, 28, 28), "reconstructions.png")

# Generate samples
with torch.no_grad():
    noise = torch.randn(batch_size, latent_dim).to(DEVICE)
    generated_images = decoder(noise)

save_image(generated_images.view(batch_size, 1, 28, 28), "generated_sample.png")
