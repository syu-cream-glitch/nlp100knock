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
print(row) # なぜnameやdtypeが表示？

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
filtered_tokens = [token for token in tokens if token in wv] # in wvはいいの？
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