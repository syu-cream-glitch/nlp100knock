import os
from dotenv import load_dotenv
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
import torch.optim as optim

train_dataset = torch.load('output/output71/output71_train.pt')
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

# 学習
epochs = 10
learning_rate = 1e-3

loss_fn = nn.BCELoss() #　バイナリクロスエントロピー
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

def train(model, dataset, embedding, loss_fn, optimizer):
    total_loss = 0
    model.train()
    size = len(dataset)
    for batch, data in enumerate(dataset):
        input_ids = data['input_ids'].to(device)
        label = data['label'].to(device).float().unsqueeze(0)

        # 平均ベクトル化
        mean_vec = mean_embedding(input_ids, embedding).unsqueeze(0)

        # 順伝播
        pred = model(mean_vec)
        loss = loss_fn(pred, label)

        # 逆伝播と最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch % 100 == 0:
            current = batch + 1
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        # エポック平均損失を計算
    avg_loss = total_loss / size
    print(f"Epoch Average Loss: {avg_loss:.6f}")

    # ファイルに書き込み（追記モード）
    with open(output_file, "a", encoding="utf-8") as output_f:
        output_f.write(f"{avg_loss}\n")

os.makedirs('output/output73', exist_ok=True)
output_file = 'output/output73/output73.txt'

for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train(model, train_dataset, embedding, loss_fn, optimizer)
print("Training complete")

# 重みとテンソルのバイアスのみ保存
torch.save(model.state_dict(), 'output/output73/logistic_regression_model.pth')
# モデルの保存
torch.save(model, 'output/output73/logistic_regression_full_model.pth')



