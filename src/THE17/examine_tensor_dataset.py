import json
from pathlib import Path

import pandas
import requests
import time
from collections import OrderedDict
from returns.result import Result, Success, Failure
from tqdm import tqdm
from typing import List, Dict

import project_utils


def generate_prompt(dataset: List[Dict], selected_dim: int, num_examples: int, source_embedding: str = 'word_embedding') -> str:
    word_value_map = {}
    for item in dataset:
        word_value_map[item['word']] = item[source_embedding][selected_dim]
    sorted_word_values = OrderedDict(sorted(word_value_map.items(), key=lambda item_inner: item_inner[1]))

    highest_values = list(sorted_word_values.items())[-num_examples:]
    lowest_values = list(sorted_word_values.items())[:num_examples]
    nearest_to_zero = sorted(word_value_map.items(), key=lambda item: abs(item[1]))[:num_examples]

    prompt = f"""We need to find an explanation for a dimension in an embedding space. 
    words that have the highest value in this dimension: {", ".join([word for word, value in highest_values])}
    words that have the lowest value in this dimension: {", ".join([word for word, value in lowest_values])}
    words that have near zero values: {", ".join([word for word, value in nearest_to_zero])}
    given these values, give a one-line explanation for what this dimension activates on. Do not say anything else, just output the answer.
    """
    return prompt


def get_response(prompt_str: str) -> Result:
    url = "http://localhost:11434/v1/chat/completions"
    request_params = {
        "messages": [{"role": "user", "content": prompt_str}],
        "model": "llama3"  # or the exact model name you're using
    }
    response = requests.post(url, json=request_params)
    if response.status_code == 200:
        model_response = response.json()
        return Success(model_response['choices'][0]['message']['content'])
    else:
        return Failure(f"Error: {response.status_code} - {response.text}")


def generate_explanations(source_file: Path,
                          output_file_prefix: str,
                          top_n: int = 10,
                          source_embedding: str = 'word_embedding') -> Path:
    formatted_data = []
    with open(source_file, 'r') as file:
        data = json.load(file)
        n_dimensions = len(data[0][source_embedding])
        for dimension in tqdm(range(n_dimensions), desc="Processing Dimensions"):
            prompt = generate_prompt(
                data,
                selected_dim=dimension,
                num_examples=top_n,
                source_embedding=source_embedding
            )

            response = get_response(prompt)
            formatted_data.append({
                "dimension": dimension,
                "explanation": response.unwrap(),
                "prompt": prompt
            })

    results_dir = project_utils.results_dir()
    output_file = results_dir / f"{output_file_prefix}_explanations_{source_embedding}_llama3_top{top_n}_{time.strftime('%m%d_%H%M%S')}.json"
    df = pandas.DataFrame(formatted_data)
    df.to_csv(results_dir / f"{output_file_prefix}_explanations_{source_embedding}_llama3_top{top_n}_{time.strftime('%m%d_%H%M%S')}.csv", index=False)
    with open(output_file, "w") as f:
        json.dump(formatted_data, f)
    return output_file


if __name__ == "__main__":
    generate_explanations(
        source_file=project_utils.results_dir() / 'the17_embeddings_dataset.json',
        output_file_prefix='the17',
        top_n=10,
        source_embedding='word_embedding'
    )
