import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
from gensim.models import KeyedVectors

# 前処理データのロード
dev_dataset = torch.load('output/output71/output71_dev.pt')

load_dotenv()
path = os.getenv('W2V_MODEL_PATH')
wv = KeyedVectors.load_word2vec_format(path, binary=True)

# nn.Embedding
vocab_size, embedding_dim = wv.vectors.shape
embedding = nn.Embedding(vocab_size, embedding_dim)
embedding.weight.data.copy_(torch.from_numpy(wv.vectors))
embedding.weight.requires_grad = False  # 埋め込み層の重みを固定

# 平均ベクトル化
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

# 占有されていたため，GPU1を使用
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
embedding = embedding.to(device)

model = LogisticRegressionModel(embedding_dim).to(device)
model.load_state_dict(torch.load('output/output73/logistic_regression_model.pth'))

# 評価
def evaluate(model, dataset, embedding):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataset:
            input_ids = data['input_ids'].to(device)
            label = data['label'].to(device).float().unsqueeze(0)

            # 平均ベクトル化
            mean_vec = mean_embedding(input_ids, embedding).unsqueeze(0)

            # 順伝播
            pred = model(mean_vec)
            predicted_label = (pred >= 0.5).float()

            total += 1
            correct += (predicted_label == label).sum().item()
    accuracy = correct / total
    return accuracy

accuracy = evaluate(model, dev_dataset, embedding)
os.makedirs('output', exist_ok=True)
output_file = 'output/output74.txt'
with open(output_file, 'w') as output_f:
    output_f.write(f'Development Set Accuracy: {accuracy:.4f}\n')

