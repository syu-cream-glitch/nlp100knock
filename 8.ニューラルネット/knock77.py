import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader

# 前処理データのロード
dev_dataset = torch.load('output/output71/output71_dev.pt')

# バッチ内のサンプルを結合する関数
def collate(batch):
    lengths = [len(x['input_ids']) for x in batch]

    sorted_idxs = sorted(range(len(lengths)), key=lambda k: lengths[k], reverse=True)
    sorted_batch = [batch[i] for i in sorted_idxs]

    max_length = max(lengths)
    input_ids_padded = []
    labels = []

    for data in sorted_batch:
        input_ids = data['input_ids']
        pad_length = max_length - len(input_ids)
        padding = torch.cat([input_ids, torch.zeros(pad_length, dtype=torch.long)])
        input_ids_padded.append(padding)
        labels.append(data['label'])
    
    input_ids_tensor = torch.stack(input_ids_padded)
    labels_tensor = torch.stack(labels)

    return {'input_ids': input_ids_tensor, 'label': labels_tensor}

# DataLoaderの作成
batch_size = 64
dev_loader = DataLoader(
    dev_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate
)

# Word2Vec読み込み
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
    return vectors.mean(dim=1)  # (batch_size, embedding_dim)

# ロジスティック回帰モデル
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# GPU使用
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
embedding = embedding.to(device)

# モデルロード
model = LogisticRegressionModel(embedding_dim).to(device)
model.load_state_dict(torch.load('output/output76/logistic_regression_model.pth'))

# 評価
def evaluate(model, data_loader, embedding):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            input_ids = data['input_ids'].to(device)
            labels = data['label'].to(device).float()  # (batch_size, 1)

            # 平均ベクトル化
            mean_vecs = mean_embedding(input_ids, embedding)  # (batch_size, embedding_dim)

            # 順伝播
            preds = model(mean_vecs)
            predicted_labels = (preds >= 0.5).float()

            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()
    accuracy = correct / total
    return accuracy

accuracy = evaluate(model, dev_loader, embedding)

# 結果出力
os.makedirs('output', exist_ok=True)
output_file = 'output/output77.txt'
with open(output_file, 'w') as output_f:
    output_f.write(f'Development Set Accuracy: {accuracy:.4f}\n')

print(f'Development Set Accuracy: {accuracy:.4f}')
