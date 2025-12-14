import os
import pandas as pd
from joblib import load
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

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

train_data = df_to_feature_list(dfs[0])
dev_data = df_to_feature_list(dfs[1])

lr_loaded, vec_loaded = load("output/output62.joblib")

X_train = vec_loaded.transform([d['feature'] for d in train_data])
X_dev = vec_loaded.transform([d['feature'] for d in dev_data])

#正解ラベルを用意
y_true_train = [int(d['label']) for d in train_data]
y_true_dev = [int(d['label']) for d in dev_data]

#各行を1サンプルとして独立に予測できる
y_pred_train = lr_loaded.predict(X_train)
y_pred_dev = lr_loaded.predict(X_dev)

#正解率，適合率（誤りを検出），再現率（見落としを検出），F1スコア（適合率と再現率を調和平均を用いて一つの評価尺度に統合）を計測．
results = {
    "train": (y_true_train, y_pred_train),
    "dev": (y_true_dev, y_pred_dev)
}

os.makedirs("output", exist_ok=True)
output_file = os.path.join("output", "output67.txt")

with open(output_file, "w", encoding="utf-8") as output_f:
    for name, (y_true, y_pred) in results.items():
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=3)
        
        output_f.write(f"=== {name.upper()} ===\n")
        output_f.write(f"Accuracy: {accuracy:.3f}\n")
        output_f.write(f"Precision: {precision:.3f}\n")
        output_f.write(f"Recall: {recall:.3f}\n")
        output_f.write(f"F1-score: {f1:.3f}\n\n")
        output_f.write("Classification Report:\n")
        output_f.write(report + "\n\n")