import os
import pandas as pd
from joblib import load
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

input_files = [
    "SST-2/train.tsv",
    "SST-2/dev.tsv"
]

dfs = [pd.read_csv(f,sep="\t") for f in input_files]

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

lr_loaded, vec_loaded = load("output/output62.joblib")
X_dev = vec_loaded.transform([d['feature'] for d in dev_data])
#正解ラベルを用意
y_true = [int(d['label']) for d in dev_data]
#各行を1サンプルとして独立に予測できる
y_pred = lr_loaded.predict(X_dev)

cm = confusion_matrix(y_true, y_pred)

#混同行列の可視化
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
display.plot(cmap="Blues")
os.makedirs("output", exist_ok=True)
output_file = os.path.join("output", "output66.png")
plt.savefig(output_file)
plt.close()

