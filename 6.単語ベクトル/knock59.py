import os
import pycountry
from dotenv import load_dotenv
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

os.makedirs("output", exist_ok=True)

countries = [country.name.replace(" ", "_") for country in pycountry.countries]

load_dotenv()
path = os.getenv("W2V_MODEL_PATH")
wv = KeyedVectors.load_word2vec_format(path, binary=True)

exist_countries = [c for c in countries if c in wv.key_to_index]
vectors =np.array([wv[c] for c in exist_countries])

tsne = TSNE(random_state=0, max_iter=15000, metric='cosine')
embs = tsne.fit_transform(vectors)
plt.figure(figsize=(10, 8))
plt.scatter(embs[:, 0], embs[:, 1])
plt.title("t-SNE visualization of country word vectors")
for (x, y), name in zip(embs, exist_countries):
    plt.text(x, y, name, fontsize=8)
plt.savefig(os.path.join("output", "tsne59.png"))
plt.show()