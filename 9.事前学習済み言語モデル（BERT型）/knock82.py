import os
import pandas as pd
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask", model="google-bert/bert-base-uncased", top_k=10
)

masked_text = "The movie was full of [MASK]."

result = fill_mask(masked_text)
df = pd.DataFrame(result)

os.makedirs("output", exist_ok=True)
output_file = os.path.join("output", "output82.txt")
with open(output_file, "w", encoding="utf-8") as output_f:
    for i in range(len(df)):
        output_f.write(f"{df['token_str'][i]}\t{df['score'][i]}\n")