"""Adapted from https://github.com/Jackson-Kang/PyTorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb.

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""

import logging
import os

import hydra
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from model import Decoder, Encoder, Model

log = logging.getLogger(__name__)


def loss_function(x, x_hat, mean, log_var):
    """Elbo loss function."""
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + kld


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main training function."""
    # Get Hydra output directory
    output_dir = HydraConfig.get().runtime.output_dir
    log.info(f"Output directory: {output_dir}")

    # Device setup
    cuda = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if cuda else "cpu")
    log.info(f"Using device: {DEVICE}")

    # Extract hyperparameters from config
    batch_size = cfg.data.batch_size
    dataset_path = cfg.data.dataset_path

    x_dim = cfg.model.x_dim
    hidden_dim = cfg.model.hidden_dim
    latent_dim = cfg.model.latent_dim

    num_epochs = cfg.training.num_epochs
    seed = cfg.training.seed

    # Set Random state for reproducibility
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    # Data loading
    mnist_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
    test_dataset = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Model setup
    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)

    model = Model(encoder=encoder, decoder=decoder).to(DEVICE)

    # Instantiate optimizer from config
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    log.info(f"Using optimizer: {optimizer.__class__.__name__}")

    # Training loop
    log.info("Start training VAE...")
    model.train()
    for epoch in range(num_epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                log.info(batch_idx)
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()
        log.info(f"Epoch {epoch + 1} complete!,  Average Loss: {overall_loss / (batch_idx * batch_size)}")
    log.info("Finish!!")

    # Save weights
    model_path = os.path.join(output_dir, "trained_model.pt")
    torch.save(model, model_path)
    log.info(f"Model saved to: {model_path}")

    # Generate reconstructions
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            if batch_idx % 100 == 0:
                log.info(batch_idx)
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)
            x_hat, _, _ = model(x)
            break

    orig_path = os.path.join(output_dir, "orig_data.png")
    recon_path = os.path.join(output_dir, "reconstructions.png")
    save_image(x.view(batch_size, 1, 28, 28), orig_path)
    save_image(x_hat.view(batch_size, 1, 28, 28), recon_path)
    log.info(f"Reconstructions saved to: {orig_path} and {recon_path}")

    # Generate samples
    with torch.no_grad():
        noise = torch.randn(batch_size, latent_dim).to(DEVICE)
        generated_images = decoder(noise)

    sample_path = os.path.join(output_dir, "generated_sample.png")
    save_image(generated_images.view(batch_size, 1, 28, 28), sample_path)
    log.info(f"Generated samples saved to: {sample_path}")


if __name__ == "__main__":
    main()
