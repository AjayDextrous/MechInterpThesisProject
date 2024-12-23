import json
from pathlib import Path

import pandas
import torch
import project_utils
from src.THE17.examine_tensor_dataset import generate_explanations


def pca(data, k):
    """
    Perform PCA on the given data and return the transformation matrix.

    Args:
        data (torch.Tensor): Input tensor of shape (n_samples, n_features).
        k (int): Number of principal components to retain.

    Returns:
        torch.Tensor: Transformation matrix of shape (n_features, k).
        torch.Tensor: Transformed data of shape (n_samples, k).
    """
    # Step 1: Compute the mean and center the data
    mean = data.mean(dim=0)
    centered_data = data - mean

    # Step 2: Compute the covariance matrix
    covariance_matrix = torch.mm(centered_data.T, centered_data) / (data.size(0) - 1)

    # Step 3: Perform eigen decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

    # Step 4: Sort eigenvalues and eigenvectors in descending order
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 5: Select the top-k eigenvectors
    transformation_matrix = eigenvectors[:, :k]

    # Step 6: Transform the data
    transformed_data = torch.mm(centered_data, transformation_matrix)

    return transformation_matrix, transformed_data


def perform_pca(source_file: Path, output_file_prefix: str, source_embedding: str = 'word_embedding', k = -1) -> Path:
    results_dir = project_utils.results_dir()
    with open(source_file, 'r') as file:
        data = json.load(file)
        embeddings = [item[source_embedding] for item in data]
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
    print(f'Embeddings : {embeddings.shape}')
    if k == -1:
        print(f"Using default value of k = {embeddings.shape[1]}")
        k = embeddings.shape[1]  # Number of principal components to retain
    transformation_matrix, transformed_data = pca(embeddings, k)

    transformed_data = transformed_data.tolist()
    print(f"data_size: {len(transformed_data[1])}")
    for i, item in enumerate(data):
        item['pca_representation'] = transformed_data[i]

    output_file_path = results_dir / f'{output_file_prefix}_pca_top{k}_embeddings_dataset.json'
    with open(output_file_path, 'w') as file:
        json.dump(data, file)

    return output_file_path


if __name__ == "__main__":
    print("Performing PCA on the17_embeddings_dataset.json")
    output_file = perform_pca(
        source_file=project_utils.results_dir() / 'the17_embeddings_dataset.json',
        output_file_prefix='the34',
        source_embedding='word_embedding',
        k=10
    )
    print("Generating explanations for PCA embeddings")
    output_file = generate_explanations(
        source_file= output_file,
        output_file_prefix='the34',
        top_n=10,
        source_embedding='pca_representation'
    )
    print(f"Explanations generated at: {output_file}")