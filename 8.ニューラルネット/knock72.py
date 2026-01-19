import os
from dotenv import load_dotenv
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
import torch.optim as optim

load_dotenv()
path = os.getenv('W2V_MODEL_PATH')
wv = KeyedVectors.load_word2vec_format(path, binary=True)

# 前処理したデータをロード
train_dataset = torch.load('output/output71/output71_train.pt')
dev_dataset = torch.load('output/output71/output71_dev.pt')

# nn.Embedding
vocab_size, embedding_dim = wv.vectors.shape
embedding = nn.Embedding(vocab_size, embedding_dim)
embedding.weight.data.copy_(torch.from_numpy(wv.vectors))
embedding.weight.requires_grad = False  # 埋め込み層の重みを固定

# 平均ベクトル
def mean_embedding(input_ids, embedding):
    vectors = embedding(input_ids)
    return vectors.mean(dim=0)

# ロジスティック回帰モデル
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegressionModel(embedding_dim)

# train_datasetの1例を使って動作確認
example = train_dataset[0]
input_ids = example['input_ids']
label = example['label']

mean_vector = mean_embedding(input_ids, embedding)
output = model(mean_vector)

os.makedirs('output', exist_ok=True)
output_file = os.path.join('output', 'output72.txt')
with open(output_file, 'w', encoding='utf-8') as output_f:
    output_f.write(f"Mean embedding vector: {mean_vector}\n")
    output_f.write(f"Model output (probability): {output.item()}\n")
    output_f.write(f"Label: {label.item()}\n")

