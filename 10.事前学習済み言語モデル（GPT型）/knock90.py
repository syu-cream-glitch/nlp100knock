import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("gpt2-medium")

prompt = "The movie was full of"

# トークン化
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]

# 推論
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    predicted_token_prob = torch.softmax(logits, dim=0)

# 上位10トークンを取得
top10_probs, top10_indices = torch.topk(predicted_token_prob, k=10)

output_file = os.path.join("output", "output90.txt")
os.makedirs("output", exist_ok=True)
with open(output_file, "w", encoding="utf-8") as output_f:
    output_f.write("トークン列の確認\n")
    for token_id in input_ids[0]:
        # 先頭の空白もトークン化されていることを確認（BPEが使用されている）
        output_f.write(f"{token_id.item()}:{repr(tokenizer.decode(token_id.item()))}\n")

    output_f.write("上位10トークンとその確率:\n")
    for prob, idx in zip(top10_probs, top10_indices):
        token = tokenizer.decode(idx.item())
        output_f.write(f"トークン: {repr(token)}, 確率: {prob.item():.4f}\n")