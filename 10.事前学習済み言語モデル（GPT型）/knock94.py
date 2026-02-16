import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")
model.eval()

chat = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "What do you call a sweet eaten after dinner?"
    },
]

prompt = tokenizer.apply_chat_template(chat, return_tensors="pt")
input_ids = prompt["input_ids"]

outputs = model.generate(
    input_ids=input_ids,
    max_length=input_ids.shape[1] + 20,
    do_sample=True,
    top_p=0.9,
    temperature=0.8,
    pad_token_id=tokenizer.eos_token_id
)

output_file = os.path.join("output", "output94.txt")
os.makedirs("output", exist_ok=True)

with open(output_file, "w", encoding="utf-8") as output_f:
    output_f.write("Prompt:\n")
    output_f.write(tokenizer.decode(input_ids[0], skip_special_tokens=True) + "\n\n")
    
    output_f.write("Generated response:\n")
    output_f.write(tokenizer.decode(outputs[0], skip_special_tokens=True) + "\n")