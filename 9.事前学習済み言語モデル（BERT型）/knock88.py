import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ファインチューニング済みモデルとトークナイザのロード
model_path = "output/output87/finetuned_bert.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-uncased"
)

# ファインチューニングを行ったモデルに合わせてロード
model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-uncased",
    num_labels=2
)
model.load_state_dict(torch.load(model_path, map_location=device)) # gpuでロード
model.to(device)
model.eval()

# 予測対象の文
sentences = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish."
]

# トークナイズ
encodings = tokenizer(
    sentences,
    padding=True,      # バッチ内で長さを揃える
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

input_ids = encodings["input_ids"].to(device)
attention_mask = encodings["attention_mask"].to(device)

# 推論
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=-1)

os.makedirs("output", exist_ok=True)
output_file = "output/output88.txt"

with open(output_file, "w", encoding="utf-8") as output_f:
    for sentence, label in zip(sentences, pred):
        sentiment = "positive" if label.item() == 1 else "negative"
        output_f.write(f"Sentence: {sentence}\nPredicted sentiment: {sentiment}\n\n")
