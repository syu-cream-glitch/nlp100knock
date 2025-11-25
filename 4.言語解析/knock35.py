import spacy
from spacy import displacy

text = "メロスは激怒した。"

nlp = spacy.load("ja_ginza")
doc = nlp(text)

displacy.serve(doc, style="dep", port=8000)
#可視化方法をREADME.mdに記載