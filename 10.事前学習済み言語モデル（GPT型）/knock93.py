import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
model.eval()

sentences = [
    "The movie was full of surprises",
    "The movies were full of surprises",
    "The movie were full of surprises",
    "The movies was full of surprises"
]

def calculate_perplexity(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # lossはクロスエントロピー
        loss = outputs.loss.item()
    
    ppl = torch.exp(torch.tensor(loss))
    return ppl.item()

output_file = os.path.join("output", "output93.txt")
os.makedirs("output", exist_ok=True)

with open(output_file, "w") as output_f:
    for s in sentences:
        ppl = calculate_perplexity(s)
        output_f.write(f"Sentence: {s}\n")
        output_f.write(f"Perplexity: {ppl:.2f}\n\n")
