import os
import pandas as pd

input_files = [
    "SST-2/train.tsv",
    "SST-2/dev.tsv"
]
os.makedirs("output", exist_ok=True)
output_file = os.path.join("output", "output60.txt")

dfs = [pd.read_csv(f, sep="\t") for f in input_files]

def Count_PandN(df):
    cntP = 0
    cntN = 0
    for _, raw in df.iterrows():
        if raw["label"] == 1:
            cntP += 1
        elif raw["label"] == 0:
            cntN += 1
    
    return cntP, cntN

with open(output_file, "w", encoding="utf-8") as output_f:
    for fname, df in zip(["train.tsv","dev.tsv"], dfs):
        cntP, cntN = Count_PandN(df)
        output_f.write(f"{fname}\npositive:{cntP}\tnegative:{cntN}\n")



