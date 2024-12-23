import json
import torch
from torch import optim
from tqdm import tqdm
import numpy as np

import project_utils
from src.models.basic_sparse_autoencoder import SparseAutoencoder

# Load your dataset
results_dir = project_utils.results_dir()
with open(results_dir / 'the17_embeddings_dataset.json', 'r') as file:
    data = json.load(file)
    embeddings = [item['word_embedding'] for item in data]
    embeddings = torch.tensor(embeddings, dtype=torch.float32)  # Your dataset as a PyTorch tensor

# Hyperparameters
input_dim = 768  # e.g., dimension of BERT embeddings
latent_dim = 768*2 #128  # Desired latent space size
l1_lambda = 1e-6  # Sparsity regularization strength

# Instantiate model
model = SparseAutoencoder(input_dim, latent_dim, l1_lambda)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 50
batch_size = 64

dataset = torch.utils.data.TensorDataset(embeddings)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in tqdm(range(num_epochs), desc="Training"):
    model.train()
    epoch_loss = 0.0

    for batch in dataloader:
        batch = batch[0]  # Unpack batch
        optimizer.zero_grad()

        # Forward pass
        reconstructed, latent = model(batch)
        print(f'latent={latent}')
        # Compute loss
        loss = SparseAutoencoder.sparse_loss(reconstructed, batch, latent, l1_lambda)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), results_dir / 'sparse_autoencoder_state_dict.pth')
model.eval()
# with torch.no_grad():
#     sparse_representations = []
#     for batch in dataloader:
#         batch = batch[0]
#         _, latent = model(batch)
#         sparse_representations.append(latent)
#     sparse_representations = torch.cat(sparse_representations, dim=0)