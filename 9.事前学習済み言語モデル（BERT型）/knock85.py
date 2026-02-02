import os
import re
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd

input_files = [
    "SST-2/train.tsv",
    "SST-2/dev.tsv"
]

# データの読み込み
dfs = [pd.read_csv(f, sep='\t') for f in input_files]
train_df, dev_df = dfs

# トークナイザーとモデルの読み込み
tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-uncased"
)

model = AutoModel.from_pretrained(
    "google-bert/bert-base-uncased"
)

# 拡張
def build_dataset(df, tokenizer, max_length=128): # SST-2は比較的短い文なので64でもいいと考える．
    dataset = []

    for _, row in df.iterrows():
        text = row['sentence']
        label = row['label']

        encodings = tokenizer(
            text,
            padding='max_length',
            truncation=True, # 文がmax_lengthを超える場合に切り捨てる
            max_length=max_length,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids'].squeeze(0) # (max_length,)
        attention_mask = encodings['attention_mask'].squeeze(0) # (max_length,)
        label_tensor = torch.tensor([float(label)], dtype=torch.float)

        data_dict = {
            "text": text,
            "label": label_tensor,
            "attention_mask": attention_mask,
            "input_ids": input_ids
        }

        dataset.append(data_dict)
    
    return dataset

train_dataset = build_dataset(train_df, tokenizer)
dev_dataset = build_dataset(dev_df, tokenizer)

# 前処理の結果を保存
os.makedirs("output/output85", exist_ok=True)
output_files = ["output85_train.pt", "output85_dev.pt"]
output_paths = [os.path.join("output/output85", f) for f in output_files]
torch.save(train_dataset, output_paths[0])
torch.save(dev_dataset, output_paths[1])

# ファイルサイズが大きいから一例のみ表示
os.makedirs('output', exist_ok=True)
output_file = os.path.join('output', 'output85.txt')
with open(output_file, 'w', encoding='utf-8') as output_f:
    output_f.write(f"before train dataset size: {len(train_df)}\n")
    output_f.write(f"before dev dataset size: {len(dev_df)}\n")
    output_f.write(f"after train dataset size: {len(train_dataset)}\n")
    output_f.write(f"after dev dataset size: {len(dev_dataset)}\n")
    output_f.write("train dataset example:\n")
    output_f.write(f"{train_dataset[0]}\n")
    output_f.write("dev dataset example:\n")
    output_f.write(f"{dev_dataset[0]}\n")