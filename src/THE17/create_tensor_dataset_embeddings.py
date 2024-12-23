from transformers import AutoTokenizer, AutoModel
from nltk.corpus import wordnet as wn
import pandas as pd
import torch
from tqdm import tqdm
import project_utils

# Load pre-trained BERT model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load all Synsets
all_synsets = list(wn.all_synsets())
formatted_data = []
not_single_token = 0
no_examples = 0
for syn in tqdm(all_synsets, desc="Processing Synsets"):
    tqdm.write(f"Word: {syn.name()} :: {syn.lemma_names()}")
    word = syn.lemma_names()[0]
    if len(syn.examples()) == 0:
        no_examples += 1
        tqdm.write(f"Skipping word '{word}' has no examples! total: {no_examples} words skipped.")
        continue
    sentence = syn.examples()[0]
    definition = syn.definition()
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True, return_offsets_mapping=True)

    device = torch.device("mps")  # Use "cpu" or "cuda" otherwise
    model = model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    tokens = tokenizer.tokenize(sentence)
    token_offsets = inputs.pop("offset_mapping")  # offsets help locate specific words in tokenized input

    # Find the position of the word

    token_index = None
    for idx, token in enumerate(tokens):
        if word in tokenizer.convert_ids_to_tokens(tokenizer(sentence)["input_ids"][idx]):
            token_index = idx
            break

    if token_index is None:
        not_single_token += 1
        tqdm.write(f"Skipping word '{word}' not found in the tokenized input! total: {not_single_token} words skipped.")
        continue
        # raise ValueError(f"Word '{word}' not found in the tokenized input!")

    # Forward pass through the model
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

    # Extract the embedding for "bank"
    word_embedding = last_hidden_state[0, token_index]  # Shape: (hidden_dim,)
    formatted_data.append(
        {
            "id": syn.name(),
            "word": word,
            "definition": definition,
            "example": sentence,
            "word_embedding": word_embedding.tolist(),
        }
    )

print("Summary :")
print("total words processed: ", len(formatted_data))
print("words without examples: ", no_examples)
print("words without single token: ", not_single_token)
# Save as CSV for easy access
df = pd.DataFrame(formatted_data)
results_dir = project_utils.results_dir()
results_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(results_dir / "the17_embeddings_dataset.csv", index=False)

# Save as JSON (optional)
import json
with open(results_dir / "the17_embeddings_dataset.json", "w") as f:
    json.dump(formatted_data, f)

print("Dataset saved in CSV, JSON formats.")