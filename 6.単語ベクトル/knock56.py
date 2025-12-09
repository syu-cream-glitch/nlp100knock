import os
from dotenv import load_dotenv
from gensim.models import KeyedVectors
import pandas as pd

load_dotenv()
path = os.getenv('W2V_MODEL_PATH')
wv = KeyedVectors.load_word2vec_format(path, binary=True)

input_file = "data/wordsim/combined.csv"
os.makedirs("output", exist_ok=True)
output_file = os.path.join("output", "output56.txt")

df = pd.read_csv(input_file)
cosine_similarity = lambda x:wv.similarity(x['Word 1'], x['Word 2'])
df['cosine_similarity'] = df.apply(cosine_similarity, axis=1)

with open(output_file, 'w', encoding='utf-8') as output_f:
    #to_string()で文字列に変換
    output_f.write(df[["Human (mean)", "cosine_similarity"]].corr(method="spearman").to_string())
