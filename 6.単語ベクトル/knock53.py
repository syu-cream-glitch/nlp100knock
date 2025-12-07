import os
from dotenv import load_dotenv
from gensim.models import KeyedVectors

load_dotenv()
path = os.getenv("W2V_MODEL_PATH")
wv = KeyedVectors.load_word2vec_format(path, binary = True)

word1 = 'Spain'
word2 = 'Madrid'
word3 = 'Athens'

#手法1
result_vector = wv[word1] - wv[word2] + wv[word3]
most_similar10 = wv.most_similar(positive=[result_vector], topn=10)

#手法2
most_similar10_2 = wv.most_similar(positive=[word1, word3], negative=[word2], topn=10)

output_file = os.path.join("output", "output53.txt")
os.makedirs("output", exist_ok=True)
with open(output_file, "w", encoding="utf-8") as output_f:
    output_f.write("手法1の結果:\n")
    for word, similarity in most_similar10:
        output_f.write(f"{word}\t{similarity}\n")
    output_f.write("\n手法2の結果:\n")
    for word, similarity in most_similar10_2:
        output_f.write(f"{word}\t{similarity}\n")
