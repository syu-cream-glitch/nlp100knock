import os
import pandas as pd
from joblib import load
from collections import Counter

input_files = [
    "SST-2/train.tsv",
    "SST-2/dev.tsv"
]
os.makedirs("output", exist_ok=True)
output_file = os.path.join("output", "output63.txt")

dfs = [pd.read_csv(f, sep="\t") for f in input_files]

def text_to_feature(text):
    tokens = text.split()
    return dict(Counter(tokens))

def df_to_feature_list(df):
    feature_list = []
    for _, row in df.iterrows():
        feature_list.append({
            'text': row['sentence'],
            'label': row['label'],
            'feature': text_to_feature(row['sentence'])
        })
    return feature_list

# 学習データはknock63では使用しない
# train_data = df_to_feature_list(dfs[0])
dev_data = df_to_feature_list(dfs[1])

#knock62で保存したファイルを読み込み
lr_loaded, vec_loaded = load("output/output62.joblib")

#BoWに変換（注意：リストしか受け取れない）
X_new = vec_loaded.transform([dev_data[0]['feature']])

#予測
y_pred = lr_loaded.predict(X_new)

with open(output_file, "w", encoding="utf-8") as output_f:
    output_f.write(f"Predicted label:{y_pred[0]}")
