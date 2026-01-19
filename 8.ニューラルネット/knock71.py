import os
import re
import torch
from dotenv import load_dotenv
from gensim.models import KeyedVectors
import pandas as pd

input_files = [
    "SST-2/train.tsv",
    "SST-2/dev.tsv"
]

# 一旦1つのデータに対して処理を試す
# データの読み込み
dfs = [pd.read_csv(f, sep='\t') for f in input_files]
train_df, dev_df = dfs
row = train_df.iloc[0]
text = row['sentence']
label = row['label']
print(row)

# 事前学習済みモデルの読み込み
load_dotenv()
path = os.getenv('W2V_MODEL_PATH')
wv = KeyedVectors.load_word2vec_format(path, binary=True)

# トークナイズ:torchtextは使用できない
def english_tokenizer(text):
    text = text.lower()
    return re.findall(r"[\w']+|[.,!?;]", text)

tokens = english_tokenizer(text)
print(tokens)

# 語彙にある単語のみ残す
filtered_tokens = [token for token in tokens if token in wv]
print(filtered_tokens)

# 単語をIDに変換
token2id = [wv.key_to_index[token] for token in filtered_tokens]
print(token2id)

# tensor化
input_ids = torch.tensor(token2id, dtype=torch.long) # 整数にはtorch.longを使用
print(input_ids)
label_tensor = torch.tensor([float(label)], dtype=torch.long) # 整数にはtorch.longを使用
print(label_tensor)

# 辞書化
example = {
    "text": text,
    "label": label_tensor,
    "input_ids": input_ids
}
print(example)

# 拡張
def build_dataset(df, wv):
    dataset = []

    for _, row in df.iterrows():
        text = row['sentence']
        label = row['label']

        tokens = english_tokenizer(text)
        filtered_tokens = [token for token in tokens if token in wv]
        if len(filtered_tokens) == 0:
            continue
        token2id = [wv.key_to_index[token] for token in filtered_tokens]

        input_ids = torch.tensor(token2id, dtype=torch.long)
        label_tensor = torch.tensor([float(label)], dtype=torch.float)

        data_dict = {
            "text": text,
            "label": label_tensor,
            "input_ids": input_ids
        }

        dataset.append(data_dict)
    
    return dataset

train_dataset = build_dataset(train_df, wv)
dev_dataset = build_dataset(dev_df, wv)

# 前処理の結果を保存
os.makedirs("output/output71", exist_ok=True)
output_files = ["output71_train.pt", "output71_dev.pt"]
output_paths = [os.path.join("output/output71", f) for f in output_files]
torch.save(train_dataset, output_paths[0])
torch.save(dev_dataset, output_paths[1])

# ファイルサイズが大きいから一例のみ表示
os.makedirs('output', exist_ok=True)
output_file = os.path.join('output', 'output71.txt')
with open(output_file, 'w', encoding='utf-8') as output_f:
    output_f.write(f"before train dataset size: {len(train_df)}\n")
    output_f.write(f"before dev dataset size: {len(dev_df)}\n")
    output_f.write(f"after train dataset size: {len(train_dataset)}\n")
    output_f.write(f"after dev dataset size: {len(dev_dataset)}\n")
    output_f.write("train dataset example:\n")
    output_f.write(f"{train_dataset[0]}\n")
    output_f.write("dev dataset example:\n")
    output_f.write(f"{dev_dataset[0]}\n")