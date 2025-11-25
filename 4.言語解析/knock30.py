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

#ja_ginza_electraは大きいモデルなので使用しなかった．(uv add ja_ginza)
nlp = spacy.load("ja_ginza")
doc = nlp(text)
verbs_list = [token.text for token in doc if token.pos_ == "VERB"]

output_file = os.path.join("output", "output30.txt")
os.makedirs("output", exist_ok = True)
with open(output_file, "w", encoding = "utf-8") as output_f:
    for verb in verbs_list:
        output_f.write(verb + "\n")

