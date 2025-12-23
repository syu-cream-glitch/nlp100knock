import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Using {} device'.format(device))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten() # 入力画像を1次元に変換，28x28 -> 784
        self.linear_relu_stack = nn.Sequential( # ネットワークの層を順番に積み重ねる
            nn.Linear(28 * 28, 512), # 784 -> 512全結合層
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )
    
    def forward(self, x): # 順伝播の定義：データがどう流れるかを定義
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits) # クラス方向（各列）
y_pred = pred_probab.argmax(1) # 最も確率の高いクラスを予測
print(f"Predicted class: {y_pred}")

# モデルレイヤー
input_image = torch.rand(3, 28, 28)
print(input_image.size())

flatten = nn.Flatten() # ミニバッチの0次元はサンプル番号をしめす次元で，nn.Flattenを通しても変化しない．
flat_image = flatten(input_image)
print(flat_image.size())

layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

print(f"Before ReLU: {hidden1}\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}\n")

# nn.Sequential：複数のレイヤーを順番に適用するコンテナ
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

# nn.softmax: ニューラルネットワークの最後のlinear layerはlogits[-∞, ∞]を出力（単純にこの範囲で出力を行うよっていう意味）．
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

# モデルのパラメータ
# モデルの各レイヤーの全てのパラメータにアクセスできるようになる．
print("Model structure:", model, "\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


