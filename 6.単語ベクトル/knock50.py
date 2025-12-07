import os
from dotenv import load_dotenv
from gensim.models import KeyedVectors

load_dotenv()
path = os.getenv('W2V_MODEL_PATH')
wv = KeyedVectors.load_word2vec_format(path, binary = True)

vector = wv['United_States']

output_file = os.path.join("output", "output50.txt")
os.makedirs("output", exist_ok = True)
with open(output_file, "w", encoding = "utf-8") as output_f:
    output_f.write(str(vector))