import project_utils
from src.THE17.examine_tensor_dataset import generate_explanations

if __name__ == "__main__":
    generate_explanations(
        source_file=project_utils.results_dir() / 'the19_embeddings_dataset.json',
        output_file_prefix='the19',
        top_n=10,
        source_embedding='latent_representation'
    )

