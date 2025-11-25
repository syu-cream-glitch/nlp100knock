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

lines = []
for token in doc:
    head_text = token.head.text if token.dep_ != "ROOT" else token.text
    lines.append(f"{token.text}\t{head_text}")

output_file = os.path.join("output", "output33.txt")
os.makedirs("output", exist_ok=True)
with open(output_file, "w", encoding="utf-8") as output_f:
    output_f.write("係り元\t係り先\n")
    output_f.writelines("\n".join(lines))