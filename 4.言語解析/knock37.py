import gzip
import re
import json
import os
import collections
import spacy

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

# GiNZA を parser/ner なしでロード
nlp = spacy.load("ja_ginza", disable=["parser","ner"])
counter = collections.Counter()
chunk_size = 40000  # Sudachi の制限

input_file = "jawiki-country.json.gz"
output_file = os.path.join("output", "output37.txt")
os.makedirs("output", exist_ok=True)

texts_to_process = []

# 記事を読み込んでチャンクごとにリストに追加
with gzip.open(input_file, "rt", encoding="utf-8") as f:
    for line in f:
        article = json.loads(line)
        text = remove_markup(article.get("text",""))
        text_bytes = text.encode("utf-8")
        for i in range(0, len(text_bytes), chunk_size):
            chunk = text_bytes[i:i+chunk_size].decode("utf-8", errors="ignore")
            texts_to_process.append(chunk)

# nlp.pipeでまとめて形態素解析
for doc in nlp.pipe(texts_to_process, batch_size=50):
    counter.update([token.lemma_ for token in doc if token.pos_=="NOUN"])

# 上位20語を出力
with open(output_file, "w", encoding="utf-8") as out_f:
    for word, count in counter.most_common(20):
        out_f.write(f"{word}\t{count}\n")
