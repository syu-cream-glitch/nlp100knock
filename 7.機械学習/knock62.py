import os
import pandas as pd
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import dump

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
# 検証データはknock62では使用しない
# dev_data = df_to_feature_list(dfs[1])

train_X_dict = [item['feature'] for item in train_data]
y_train = [int(item['label']) for item in train_data]

#疎行列にして非ゼロを保存→メモリと計算効率を担保
vec = DictVectorizer()
X_train = vec.fit_transform(train_X_dict)

#max_iter:収束しやすくする，C:正則化の強さ（デフォルト），solver:L2正則化しよう（デフォルト），mulit_class:2クラス分類なため，なんでもいい
lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='ovr')
lr.fit(X_train, y_train)

#予測に使用するために学習済みlrとvecを保存
output_file = os.path.join("output", "output62.joblib")
os.makedirs("output", exist_ok=True)
dump((lr,vec), output_file)


