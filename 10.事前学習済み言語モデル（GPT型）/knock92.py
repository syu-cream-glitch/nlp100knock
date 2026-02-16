import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
model.eval()

prompt = "The movie was full of"

# トークン化
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]

output_file = os.path.join("output", "output92.txt")
os.makedirs("output", exist_ok=True)

with open(output_file, "w", encoding="utf-8") as output_f:
    with torch.no_grad():
        generated_ids = input_ids.clone()
        
        output_f.write(f"Prompt: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}\n\n")
        output_f.write(f"Method: greedy\n")
        output_f.write("Token probabilities:\n")
        
        max_length = 30
        for _ in range(max_length - input_ids.shape[1]):
            outputs = model(generated_ids)
            logits = outputs.logits
            next_token_logits = logits[0, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # 次のトークンを選択
            next_token_id = torch.argmax(probs).unsqueeze(0)
            prob_next_token = probs[next_token_id].item()
            
            # トークン文字列に変換
            token_str = tokenizer.decode(next_token_id)
            output_f.write(f"Token: {repr(token_str):<12}  Prob: {prob_next_token:.4f}\n")
            
            # 生成列に追加
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)

    final_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    output_f.write("\nGenerated text:\n")
    output_f.write(final_text + "\n")
    