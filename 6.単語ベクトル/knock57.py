import pycountry
import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from gensim.models import KeyedVectors

load_dotenv()
path = os.getenv('W2V_MODEL_PATH')
wv = KeyedVectors.load_word2vec_format(path, binary=True)

os.makedirs("output", exist_ok=True)
country_file = os.path.join("output", "country57.csv")
output_file = os.path.join("output", "output57.txt")

# 国名リストを取得してCSVに保存（保存したcsvは別に使用しない）
countries = [country.name.replace(" ", "_") for country in pycountry.countries]
df = pd.DataFrame(countries, columns = ['country_name'])
df.to_csv(country_file, index=False, encoding='utf-8-sig')

#単語の辞書キーを参照（単語ベクトルが存在するものだけ取ってくる）
exist_countries = [c for c in countries if c in wv.key_to_index]

vectors = np.array([wv[c] for c in exist_countries])
km = KMeans(n_clusters=5, random_state=0)
labels = km.fit_predict(vectors)

with open(output_file, 'w', encoding='utf-8-sig') as output_f:
    for label, country in zip(labels, exist_countries):
        output_f.write(f"{label}\t{country}\n")


