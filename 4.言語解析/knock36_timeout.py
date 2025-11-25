import gzip
import shutil
import re
import json
import os
import collections
import spacy

with gzip.open("jawiki-country.json.gz", "rt", encoding = "utf-8")as input_file:
    with open("jawiki-country.json", "wt", encoding = "utf-8") as output_file:
        shutil.copyfileobj(input_file, output_file)

def remove_markup(text):
        text = re.sub("'''''", "", text)
        text = re.sub("'''", "", text)
        text = re.sub("''", "", text)
        text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
        text = re.sub(r"\{\{(?:[^|\}]*\|)?([^\}]+)\}\}", r"\1", text)
        text = re.sub(r"<ref.*?/>", "", text)
        text = re.sub(r"<ref.*?>.*?</ref>", "", text)
        text = re.sub(r"\<br\s+\/\>", "", text)
        return text

before_list = []
after_list = []
word_list = []
nlp = spacy.load("ja_ginza")

with open("jawiki-country.json", "r", encoding = "utf-8") as input_f, \
    open("output/knock36.txt", "w", encoding = "utf-8") as output_f:
    lines = input_f.readlines()
    for line in lines:
        before_list.append(json.loads(line))
    for item in before_list:
        after_list.append(remove_markup(item['text']))
    
    for text in after_list:
        chunk_size = 40000  # 1記事あたり約49KB以上のテキストは処理できない
        for i in range(0, len(text.encode('utf-8')), chunk_size):
            chunk = text.encode('utf-8')[i:i+chunk_size].decode('utf-8', errors='ignore')
            doc = nlp(chunk)
            for token in doc:
                word_list.append(token.lemma_)
    counter = collections.Counter(word_list)
    for word, count in counter.most_common(20):
        output_f.write(f"{word}\t{count}\n")
        
    




