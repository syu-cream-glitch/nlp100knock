import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

training_dataLoader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataLoader = DataLoader(test_data, batch_size=64, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

learning_rate = 1e-3 # 学習率：値が小さいと変化が少なく、大きすぎると訓練に失敗する可能性が生まれる
batch_size = 64 # バッチサイズ：ミニバッチサイズを構成するデータ数
epochs = 20 # エポック数：イテレーション回数

# 1エポック：訓練ループと検証/テストループによって構成される．
# 訓練ループ：データセットに対して訓練を実行し，パラメータを収束させる．
# 検証/テストループ：テストデータセットでモデルを評価し，性能が向上しているかを確認する．
# 回帰タスク：nn.MSELoss()，分類タスク：nn.NNLLoss()などを用いる．なお，nn.CrossEntropyLoss()は，それぞれを結合した損失関数．

loss_fn = nn.CrossEntropyLoss()

# optimizer：モデルパラメータを調節するプロセス．
# optimization algorithm：最適化プロセスの具体的な手続き．
optimizar = torch.optim.SGD(model.parameters(), lr=learning_rate)

# [1] optimizer.zero_grad()を実行し、モデルパラメータの勾配をリセットします。
# 勾配の計算は蓄積されていくので、毎イテレーション、明示的にリセットします。
# [2] 続いて、loss.backwards()を実行し、バックプロパゲーションを実行します。
# PyTorchは損失に対する各パラメータの偏微分の値（勾配）を求めます。
#[3] 最後に、optimizer.step()を実行し、各パラメータの勾配を使用してパラメータの値を調整します。

# 実装全体
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 予測と損失の計算
        pred = model(X)
        loss = loss_fn(pred, y)

        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]") # 1エポック当たり約938バッチだから，0,100,200,...900まで表示される

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()  # これでDropoutやBatchNormが評価モードになる．このコードではNeuralNetworkを特にいじっていないため必要ない．
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() # pred.argmax(1)：一番確率が高いクラスを予測
    
    test_loss /= size # 平均損失を計算
    correct /= size # 正解率を計算
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(training_dataLoader, model, loss_fn, optimizar) # 実体を呼び出すから，繰り返すだけで更新が進む．
    test_loop(test_dataLoader, model, loss_fn)
print("Done!")

# 豆知識：uvx nvitopでGPU使用率を確認できる．