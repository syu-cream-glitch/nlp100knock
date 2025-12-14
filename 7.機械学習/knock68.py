import os
import numpy as np
from joblib import load

lr_loaded, vec_loaded = load("output/output62.joblib")

#特徴量の名前
feature_names = np.array(vec_loaded.get_feature_names_out())

#2クラス分類なので lr.coef_ は shape=(1, n_features)
weights = lr_loaded.coef_[0]

# 上位20の重み（argsort()はソート後の順番に対応する「元のインデックス」を返す）
top20_idx = weights.argsort()[-20:][::-1]
top20_features = feature_names[top20_idx]
top20_weights = weights[top20_idx]

# 下位20の重み
bottom20_idx = weights.argsort()[:20]
bottom20_features = feature_names[bottom20_idx]
bottom20_weights = weights[bottom20_idx]

os.makedirs("output", exist_ok=True)
output_file = os.path.join("output", "output68.txt")

with open(output_file, "w", encoding="utf-8") as output_f:
    output_f.write("=== 重みが高い特徴量トップ20 ===\n")
    for feature, weight in zip(top20_features, top20_weights):
        output_f.write(f"{feature}: {weight:.4f}\n")
    
    output_f.write("\n=== 重みが低い特徴量トップ20 ===\n")
    for feature, weight in zip(bottom20_features, bottom20_weights):
        output_f.write(f"{feature}: {weight:.4f}\n")
