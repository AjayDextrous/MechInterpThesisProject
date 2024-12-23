import json
import pandas as pd
import project_utils
results_dir = project_utils.results_dir()
with open(results_dir / 'the19_explanations_llama3_top10_1222_184544.json', 'r') as file:
    data = json.load(file)
    df = pd.DataFrame(data)
    df.to_csv(results_dir / "the19_explanations_llama3_top10_1222_184544.csv", index=False)
