import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader

# 前処理データのロード
train_dataset = torch.load('output/output71/output71_train.pt')
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
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate
)

load_dotenv()
path = os.getenv('W2V_MODEL_PATH')
wv = KeyedVectors.load_word2vec_format(path, binary=True)

# nn.Embedding
vocab_size, embedding_dim = wv.vectors.shape
embedding = nn.Embedding(vocab_size, embedding_dim)
embedding.weight.data.copy_(torch.from_numpy(wv.vectors))
embedding.weight.requires_grad = False

# 練習がてらGPU1を使用
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
embedding = embedding.to(device)

# BiGRUモデル
class BiGRUClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.bigru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x: (batch_size, seq_len, embedding_dim)
        _, h = self.bigru(x)
        # h: (num_layers * num_directions, batch, hidden_dim)
        # num_directions=2→h[0], h[1]
        h_forward = h[0]
        h_backward = h[1]
        h_cat = torch.cat([h_forward, h_backward], dim=1)
        return torch.sigmoid(self.fc(h_cat))


epochs = 10
learning_rate = 1e-3
hidden_dim = 128

model = BiGRUClassifier(embedding_dim, hidden_dim).to(device)

loss_fn = nn.BCELoss() #　バイナリクロスエントロピー
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # SGDからAdamに変更

# 学習
def train(model, data_loader, embedding, loss_fn, optimizer):
    total_loss = 0
    model.train()
    num_batches = len(data_loader)
    for batch, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)
        labels = data['label'].to(device).float()  # (batch_size, 1), 今回は0,1ラベルだから簡単．

        # 平均ベクトル化しない
        emb = embedding(input_ids)

        # 順伝播
        preds = model(emb)
        loss = loss_fn(preds, labels)

        # 逆伝播と最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch % 100 == 0:
            print(f"loss: {loss.item():.6f}  [{batch + 1}/{num_batches}]")
    
    # エポック平均損失を計算
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch Average Loss: {avg_loss:.6f}")

    # ファイルに書き込み（追記モード）
    with open(output_file, "a", encoding="utf-8") as output_f:
        output_f.write(f"{avg_loss}\n")

os.makedirs('output/output79', exist_ok=True)
output_file = 'output/output79/output79_epoch_loss.txt'

for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train(model, train_loader, embedding, loss_fn, optimizer)
print("Training complete")

# 重みとテンソルのバイアスのみ保存
torch.save(model.state_dict(), 'output/output79/bigruc_classifier_model.pth')
# モデルの保存
torch.save(model, 'output/output79/bigruc_classifier_full_model.pth')

# モデルロード
model = BiGRUClassifier(embedding_dim, hidden_dim).to(device)
model.load_state_dict(torch.load('output/output79/bigruc_classifier_model.pth'))

# 評価
def evaluate(model, data_loader, embedding):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            input_ids = data['input_ids'].to(device)
            labels = data['label'].to(device).float()  # (batch_size, 1)

            # 平均ベクトル化しない
            emb = embedding(input_ids)

            # 順伝播
            preds = model(emb)
            predicted_labels = (preds >= 0.5).float()

            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()
    accuracy = correct / total
    return accuracy

accuracy = evaluate(model, dev_loader, embedding)

# 結果出力
os.makedirs('output', exist_ok=True)
output_file = 'output/output79/output79_accuracy.txt'
with open(output_file, 'w') as output_f:
    output_f.write(f'Development Set Accuracy: {accuracy:.4f}\n')

print(f'Development Set Accuracy: {accuracy:.4f}')