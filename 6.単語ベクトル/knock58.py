import os
import pycountry
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from gensim.models import KeyedVectors
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

os.makedirs("output", exist_ok=True)

countries = [country.name.replace(" ", "_") for country in pycountry.countries]

load_dotenv()
path = os.getenv('W2V_MODEL_PATH')
wv = KeyedVectors.load_word2vec_format(path, binary=True)

exist_countries = [c for c in countries if c in wv.key_to_index]
vectors = np.array([wv[c] for c in exist_countries])

fig = plt.figure(figsize=(30, 20))
linkage_result = linkage(vectors, method='ward')
dg = dendrogram(linkage_result, labels = exist_countries)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()
fig.savefig(os.path.join("output", "dendrogram58.png"))
