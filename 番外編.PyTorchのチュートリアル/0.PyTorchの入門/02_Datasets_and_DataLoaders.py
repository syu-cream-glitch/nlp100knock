import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
import os

# データセットの読み込み
# root：訓練/テストデータが格納されているパスを指定
# train：訓練データまたはテストデータセットを指定
# download=True：rootにデータが存在しない場合は，インターネットからデータをダウンロード
# transformとtarget_transform：特徴量とラベルの変換を指定
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

# データセットの反復処理と可視化
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item() # len(training_data)：データ数60000，torch.randint：0-59999のランダムな整数を一つ生成，
    img, label = training_data[sample_idx] # Tensor, shape: [1, 28, 28]なお、1はチャネル数
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

os.makedirs("img_fashion", exist_ok=True)
plt.savefig("img_fashion/fashion_mnist.png")
plt.close()