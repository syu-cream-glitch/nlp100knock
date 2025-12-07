import os
from dotenv import load_dotenv
from gensim.models import KeyedVectors

load_dotenv()
path = os.getenv('W2V_MODEL_PATH')
wv = KeyedVectors.load_word2vec_format(path, binary = True)
w1 = 'United_States'
w2 = 'U.S.'

cosine_similarity = wv.similarity(w1, w2)

output_file = os.path.join("output", "output51.txt")
os.makedirs("output", exist_ok = True)
with open(output_file, "w", encoding = "utf-8") as output_f:
    output_f.write(str(cosine_similarity))