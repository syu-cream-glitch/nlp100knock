import gzip
import json
import math
import collections
import spacy
import re
import os

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

# parser と ner を無効化して高速化
nlp = spacy.load("ja_ginza", disable=["parser", "ner"])

input_file = "jawiki-country.json.gz"
os.makedirs("output", exist_ok=True)
output_file = os.path.join("output", "output38.txt")

chunk_size = 40000  # Sudachi の制限
texts_to_process = []       # 解析用チャンクのリスト
titles_to_process = []      # 元記事のタイトルを記録
total_docs = 0

# 1. 記事を読み込んでチャンクに分割し、リストに追加
with gzip.open(input_file, "rt", encoding="utf-8") as input_f:
    for line in input_f:
        total_docs += 1
        article = json.loads(line)
        text = remove_markup(article.get("text", ""))
        text_bytes = text.encode("utf-8")
        for i in range(0, len(text_bytes), chunk_size):
            chunk = text_bytes[i:i+chunk_size].decode("utf-8", errors="ignore")
            texts_to_process.append(chunk)
            titles_to_process.append(article.get("title"))

# 2. 文書頻度・日本記事のTFを計算
doc_freq = collections.Counter()
japan_noun_freq = collections.Counter()

# nlp.pipeでまとめて形態素解析 → 高速化
for doc, title in zip(nlp.pipe(texts_to_process, batch_size=200), titles_to_process):
    doc_nouns = set()
    for token in doc:
        if token.pos_ == "NOUN":
            noun = token.lemma_
            doc_nouns.add(noun)
            if title == "日本":
                japan_noun_freq[noun] += 1
    for noun in doc_nouns:
        doc_freq[noun] += 1

# 3. TF-IDF計算
tfidf_scores = {}
for noun, tf in japan_noun_freq.items():
    idf = math.log(total_docs / doc_freq[noun])
    tfidf_scores[noun] = {"tf": tf, "idf": idf, "tfidf": tf * idf}

# 4. 上位20語をファイル出力
with open(output_file, "w", encoding="utf-8") as out_f:
    out_f.write("名詞\tTF\tIDF\tTF-IDF\n")
    top20 = sorted(tfidf_scores.items(), key=lambda x: x[1]["tfidf"], reverse=True)[:20]
    for noun, scores in top20:
        out_f.write(f"{noun}\t{scores['tf']}\t{scores['idf']:.4f}\t{scores['tfidf']:.4f}\n")
