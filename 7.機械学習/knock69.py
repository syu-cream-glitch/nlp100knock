import os
import pandas as pd
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

input_files = [
    "SST-2/train.tsv",
    "SST-2/dev.tsv"
]

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

train_data = df_to_feature_list(dfs[0])
dev_data = df_to_feature_list(dfs[1])

train_X_dict = [d['feature'] for d in train_data]
dev_X_dict = [d['feature'] for d in dev_data]

y_train = [int(d['label']) for d in train_data]
y_dev = [int(d['label']) for d in dev_data]


#疎行列にして非ゼロを保存→メモリと計算効率を担保
vec = DictVectorizer()
X_train = vec.fit_transform(train_X_dict)
X_dev = vec.transform(dev_X_dict)

# 正則化パラメータをリストで指定
C_values = [0.01, 0.1, 1, 10, 100]
accuracy_list = []

for C in C_values:
    lr = LogisticRegression(max_iter=1000, C=C, solver='lbfgs', multi_class='ovr')
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_dev)
    acc = accuracy_score(y_dev, y_pred)
    accuracy_list.append(acc)

os.makedirs("output", exist_ok=True)
output_file = os.path.join("output", "output69.txt")
with open(output_file, "w", encoding="utf-8") as output_f:
    for C, acc in zip(C_values, accuracy_list):
        output_f.write(f"C={C}: accuracy={acc:.4f}\n")

# グラフ描画
plt.figure(figsize=(6,4))
plt.plot(C_values, accuracy_list, marker='o')
plt.xscale('log')  # 対数スケール
plt.xlabel("Regularization parameter C")
plt.ylabel("Accuracy on dev set")
plt.title("Effect of regularization on dev accuracy")
plt.grid(True)
plt.savefig("output/output69.png")
plt.show()



