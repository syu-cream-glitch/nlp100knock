import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# GPU設定
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
model.to(device)

def make_fewshot_prompt(text):
    return f"""
Classify the sentiment of the following sentences.
Only reply with a single word: "positive" or "negative".

Examples:
Sentence: I love this movie! It's amazing.
Answer: positive

Sentence: The plot was boring and predictable.
Answer: negative

Sentence: The acting was fantastic, I enjoyed it a lot.
Answer: positive

Sentence: I did not like the food at all.
Answer: negative

Now classify this sentence:
Sentence: {text}
Answer:
"""

def predict_sentiment(text):
    prompt = make_fewshot_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # 長さの保持
    input_length = inputs.input_ids.shape[1]

    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=3,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_tokens = outputs[0][input_length:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()

    if "positive" in answer:
        return 1
    elif "negative" in answer:
        return 0
    else:
        return 0

# SST-2データセットの読み込み
dev_path = "SST-2/dev.tsv"
dataset = pd.read_csv(dev_path, sep="\t", header=0)

output_file = os.path.join("output", "output96.txt")
os.makedirs("output", exist_ok=True)
with open(output_file, "w", encoding="utf-8") as output_f:
    correct = 0
    for text, label in zip(dataset["sentence"], dataset["label"]):
        pred = predict_sentiment(text)
        if pred == label:
            correct += 1
        
    accuracy = correct / len(dataset)
    print(f"Accuracy on SST-2 dev: {accuracy:.3f}")
    output_f.write(f"Accuracy on SST-2 dev: {accuracy:.3f}\n")