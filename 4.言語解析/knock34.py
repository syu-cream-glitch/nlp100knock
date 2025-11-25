import spacy
import os

text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

nlp = spacy.load("ja_ginza")
doc = nlp(text)

results_list = []
for token in doc:
    if token.dep_ == "nsubj" and token.text == "メロス":
       subject = token.head
       results_list.append(f"{token.text}\t{subject.text}") 

output_file = os.path.join("output", "output34.txt")
os.makedirs("output", exist_ok=True)
with open(output_file, "w", encoding="utf-8") as output_f:
    for result in results_list:
        output_f.write(result + "\n")
                