import json
import torch
import project_utils
from src.models.basic_sparse_autoencoder import SparseAutoencoder
import torch.nn.functional as F

# Load your dataset
results_dir = project_utils.results_dir()
with open(results_dir / 'the17_embeddings_dataset.json', 'r') as file:
    data = json.load(file)

# Hyperparameters
input_dim = 768  # e.g., dimension of BERT embeddings
latent_dim = 768*2  # Desired latent space size
l1_lambda = 1e-6  # Sparsity regularization strength

# Instantiate model
model = SparseAutoencoder(input_dim, latent_dim, l1_lambda)
model.load_state_dict(torch.load(results_dir / 'sparse_autoencoder_state_dict.pth'))
model.eval()

# Get latent representations
for item in data:
    embedding = torch.tensor(item['word_embedding'], dtype=torch.float32)
    with torch.no_grad():
        reconstructed, latent = model(embedding)
        loss = F.mse_loss(reconstructed, embedding)
        print(f"latent: {latent.sum().item()}, recon: {reconstructed.sum().item()}")
    item['latent_representation'] = latent.tolist()

with open(results_dir / 'the19_embeddings_dataset.json', 'w') as file:
    json.dump(data, file)
