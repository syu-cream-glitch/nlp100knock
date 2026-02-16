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


# デコーディング方法と温度パラメータの設定
configs = [
    {"method": "greedy", "do_sample": False},
    {"method": "sample_temp_0.7", "do_sample": True, "temperature": 0.7},
    {"method": "sample_temp_1.2", "do_sample": True, "temperature": 1.2},
    {"method": "top_k_50", "do_sample": True, "temperature": 1.0, "top_k": 50},
    {"method": "top_p_0.9", "do_sample": True, "temperature": 1.0, "top_p": 0.9},
    {"method": "beam_search", "do_sample": False, "num_beams": 5}
]

output_file = os.path.join("output", "output91.txt")
os.makedirs("output", exist_ok=True)

with open(output_file, "w", encoding="utf-8") as f:
    for config in configs:
        method_name = config["method"]

        # generate用引数をコピー
        gen_kwargs = {k: v for k, v in config.items() if k != "method"}

        # 共通設定を追加
        gen_kwargs["max_length"] = 30

        with torch.no_grad():
            output = model.generate(input_ids, **gen_kwargs)

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        f.write(f"{method_name}: {generated_text}\n\n")

