import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding # llm1（p95参照）

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
    padding='longest', # バッチ内で最長のものに合わせてパディング
    return_tensors='pt'
)

batch_size = 4

# DataLoaderの作成
# 教科書ならbatch_inputs = data_collator()
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

os.makedirs("output", exist_ok=True)
output_file = os.path.join("output", "output86.txt")

# 最初のミニバッチを取得
batch = next(iter(train_dataloader))

with open(output_file, "w", encoding="utf-8") as output_f:
    output_f.write(f"batch inputs shape: {batch['input_ids'].shape}\n")
    output_f.write(f"batch attention_mask shape: {batch['attention_mask'].shape}\n")
    # bertが扱える形状に
    output_f.write(f"batch labels shape: {batch['labels'].squeeze(-1).shape}\n")
    