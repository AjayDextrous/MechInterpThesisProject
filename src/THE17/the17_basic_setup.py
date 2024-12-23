from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"  # You can choose other models like 'distilbert-base-uncased' for lighter computation
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Input sentence
sentence = "The river bank near my house"

# Tokenize input and find the token index for "bank"
inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True, return_offsets_mapping=True)
tokens = tokenizer.tokenize(sentence)
token_offsets = inputs.pop("offset_mapping")  # offsets help locate specific words in tokenized input

# Find the position of the word "bank"
word = "bank"
token_index = None
for idx, token in enumerate(tokens):
    if word in tokenizer.convert_ids_to_tokens(tokenizer(sentence)["input_ids"][idx]):
        token_index = idx
        break

if token_index is None:
    raise ValueError(f"Word '{word}' not found in the tokenized input!")

# Forward pass through the model
outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

# Extract the embedding for "bank"
word_embedding = last_hidden_state[0, token_index]  # Shape: (hidden_dim,)

print(f"Embedding for the word '{word}': {word_embedding}")
