import os
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-uncased"
)

result = tokenizer.tokenize("The movie was full of incomprehensibilities.")

os.makedirs("output", exist_ok=True)
output_file = os.path.join("output", "output80.txt")

with open(output_file, "w", encoding="utf-8") as output_f:
    output_f.write(str(result))

