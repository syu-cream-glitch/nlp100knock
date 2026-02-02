import os
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm

train_dataset_path = "output/output85/output85_train.pt"
dev_dataset_path = "output/output85/output85_dev.pt"

# データセットの読み込み
train_dataset = torch.load(train_dataset_path)
dev_dataset = torch.load(dev_dataset_path)

# label→labelsに変更，textもpopで削除（BatchEncodingを使用していたらこんなことにはならなかった．）
for dataset in [train_dataset, dev_dataset]:
    for data in dataset:
        data["labels"] = data.pop("label").long()
        if "text" in data:
            data.pop("text")

# トークナイザのロード
tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-uncased"
)

# DataCollatorWithPaddingの準備
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding='longest', # バッチ内で最長のものに合わせてパ
    return_tensors='pt'
)

batch_size = 128

# DataLoaderの作成
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator
)

dev_dataloader = DataLoader(
    dev_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=data_collator
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルのロード（分類用に調整）
model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-uncased",
    num_labels=2
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
# 10で遅すぎたため，3に変更
num_epochs = 3

os.makedirs("output/output87", exist_ok=True)
log_file = os.path.join("output/output87", "output87.txt")
model_file = os.path.join("output/output87", "finetuned_bert.pt")

# ファインチューニング
with open(log_file, "w", encoding="utf-8") as log_f:
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].squeeze(-1).to(device)  # 1次元に変換

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        log_f.write(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}\n")

        # 検証
        model.eval()
        correct = 0
        total = 0
        eval_total_loss = 0

        with torch.no_grad():
            for batch in dev_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].squeeze(-1).to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                eval_total_loss += loss.item() * labels.size(0) # バッチサイズで重みづけ

                pred = torch.argmax(outputs.logits, dim=-1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        val_loss = eval_total_loss / total
        accuracy = correct / total
        
        print(f"Validation Loss: {val_loss:.4f} | Accuracy: {accuracy:.4f}")
        log_f.write(f"Validation Loss: {val_loss:.4f} | Accuracy: {accuracy:.4f}\n")

# モデルの保存
torch.save(model.state_dict(), model_file)
    