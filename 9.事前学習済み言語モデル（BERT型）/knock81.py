import os
import pandas as pd
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask", model="google-bert/bert-base-uncased", top_k=1
)

masked_text = "The movie was full of [MASK]."

result = fill_mask(masked_text)

os.makedirs("output", exist_ok=True)
output_file = os.path.join("output", "output81.txt")
with open(output_file, "w", encoding="utf-8") as output_f:
    output_f.write(pd.DataFrame(result)['token_str'][0])