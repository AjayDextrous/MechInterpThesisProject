from typing import List

import torch
import torch.nn as nn
import torch.optim as optim


class MultiLayerSparseAutoencoder(nn.Module):
    def __init__(self, dimensions: List[int], l1_lambda):
        super(MultiLayerSparseAutoencoder, self).__init__()

        encoder_layers = []
        for dimension in range(len(dimensions) - 1):
            encoder_layers.append(nn.Linear(dimensions[dimension], dimensions[dimension + 1]))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for dimension in range(len(dimensions) - 1, 0, -1):
            decoder_layers.append(nn.Linear(dimensions[dimension], dimensions[dimension - 1]))
            decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

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
