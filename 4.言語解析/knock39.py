import gzip
import json
import re
import collections
import spacy
import matplotlib.pyplot as plt
import os

# -------------------------
# マークアップ除去関数
# -------------------------
def remove_markup(text):
    text = re.sub("'''''", "", text)
    text = re.sub("'''", "", text)
    text = re.sub("''", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\{\{(?:[^|\}]*\|)?([^\}]+)\}\}", r"\1", text)
    text = re.sub(r"<ref.*?/>", "", text)
    text = re.sub(r"<ref.*?>.*?</ref>", "", text)
    text = re.sub(r"<br\s*/>", "", text)
    return text

# -------------------------
# 設定
# -------------------------
input_file = "jawiki-country.json.gz"
output_file = os.path.join("output", "output39.txt")
os.makedirs("output", exist_ok=True)
chunk_size = 40000  # Sudachi の制限

# parser/ner 無効化で高速化
nlp = spacy.load("ja_ginza", disable=["parser", "ner"])
counter = collections.Counter()
texts_to_process = []

# -------------------------
# 記事をチャンクごとにリストに追加
# -------------------------
with gzip.open(input_file, "rt", encoding="utf-8") as f:
    for line in f:
        article = json.loads(line)
        text = remove_markup(article.get("text",""))
        text_bytes = text.encode("utf-8")
        for i in range(0, len(text_bytes), chunk_size):
            chunk = text_bytes[i:i+chunk_size].decode("utf-8", errors="ignore")
            texts_to_process.append(chunk)

# -------------------------
# nlp.pipeでまとめて解析（高速化）
# -------------------------
for doc in nlp.pipe(texts_to_process, batch_size=50):
    counter.update([token.lemma_ for token in doc if token.is_alpha])

# -------------------------
# 上位20語をファイル出力
# -------------------------
with open(output_file, "w", encoding="utf-8") as output_f:
    for word, count in counter.most_common(20):
        output_f.write(f"{word}\t{count}\n")

# -------------------------
# 両対数グラフ描画
# -------------------------
freqs = [count for word, count in counter.most_common()]
ranks = range(1, len(freqs)+1)

plt.figure(figsize=(8,6))
plt.loglog(ranks, freqs, marker="o", linestyle="none")
plt.xlabel("Rank (log)")
plt.ylabel("Frequency (log)")
plt.title("Word Frequency vs Rank (Log-Log)")
plt.grid(True, which="both", ls="--")
plt.savefig(os.path.join("output", "word_frequency_rank.png"))
