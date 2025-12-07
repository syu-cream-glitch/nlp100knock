import os
from dotenv import load_dotenv
from gensim.models import KeyedVectors

load_dotenv()
path = os.getenv('W2V_MODEL_PATH')
wv = KeyedVectors.load_word2vec_format(path, binary = True)

"""
topn (int or None, optional) – Number of top-N similar keys to return, 
when topn is int. When topn is None, then similarities for all keys are returned.
公式ドキュメント参照
"""

most_similar10 = wv.most_similar(positive=['United_States'], topn=10)

output_file = os.path.join("output", "output52.txt")
os.makedirs("output", exist_ok = True)
with open(output_file, "w", encoding = "utf-8") as output_f:
    for word, similarity in most_similar10:
        output_f.write(f"{word}\t{similarity}\n")