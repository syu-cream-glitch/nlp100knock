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


noun_phrases = []
for token in doc:
    #token.pos_ = "ADP"は名詞句に対応できなさそう．
    if token.text == "の" and token.dep_ == "case":
        left_word = token.head
        right_word = token.head.head
        if left_word.pos_ in ["NOUN", "PROPN"] and right_word.pos_ in ["NOUN", "PROPN"]:
            noun_phrase = left_word.text + token.text + right_word.text
            noun_phrases.append(noun_phrase)

output_file = os.path.join("output", "output32.txt")
os.makedirs("output", exist_ok = True)
with open(output_file, "w", encoding = "utf-8") as output_f:
    for word in noun_phrases:
        output_f.write(word + "\n") 