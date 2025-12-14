import os
from joblib import load
from collections import Counter

text = "the worst movie I 've ever seen"

def text_to_feature(text):
    tokens = text.split()
    return dict(Counter(tokens))

lr_loaded, vec_loaded = load("output/output62.joblib")
X_new = vec_loaded.transform([text_to_feature(text)])
y_pred = lr_loaded.predict(X_new)

os.makedirs("output", exist_ok=True)
output_file = os.path.join("output", "output65.txt")
with open(output_file, "w", encoding="utf-8") as output_f:
    output_f.write(f"Predicted label:{y_pred[0]}")


