import os
import json
import pandas as pd
from collections import Counter

input_files = [
    "SST-2/train.tsv",
    "SST-2/dev.tsv"
]
os.makedirs("output/output61", exist_ok=True)
files = ["output61_train.json", "output61_dev.json"]
output_files = [os.path.join("output/output61", f) for f in files]

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

for df, output_file in zip(dfs, output_files):
    feature_list = df_to_feature_list(df)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(feature_list, f, ensure_ascii=False, indent=2)     



