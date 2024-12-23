import torch
import torch.nn as nn
import torch.optim as optim


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, l1_lambda):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.l1_lambda = l1_lambda

    def forward(self, x):
        z = torch.relu(self.encoder(x))  # Sparse latent space
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z

    @staticmethod
    def sparse_loss(reconstructed, original, latent, l1_lambda):
        mse_loss = nn.MSELoss()(reconstructed, original)  # Reconstruction loss
        l1_loss = l1_lambda * torch.sum(torch.abs(latent))  # Sparsity penalty
        return mse_loss + l1_loss
