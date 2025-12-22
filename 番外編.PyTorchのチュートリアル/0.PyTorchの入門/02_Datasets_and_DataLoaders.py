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
    plt.imshow(img.squeeze(), cmap="gray") # img.squeeze()：Tensorの形状を[28, 28]に変換

os.makedirs("img_fashion", exist_ok=True)
plt.savefig("img_fashion/fashion_mnist.png")
plt.close()

# カスタムデータセットの作成
# 必須関数__init__, __len__, __getitem__

import pandas as pd
from torchvision.io import read_image # 画像ファイルをTensorに変換して読み込む

class CustomImageDataset(Dataset): # 独自のデータセットクラスを定義：from torch.utils.data import Datasetが必要，バッチ処理やシャッフルが可能になる．
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None): # tranform：画像データをモデルが扱いやすい形に変換，target_transform：ラベルをモデルが扱いやすい形に変換．
        self.img_labels = pd.read_csv(annotations_file) # self.img_labels.iloc[idx, 0]：i番目の画像名，self.img_labels.iloc[idx, 1]：i番目のラベルを取得できるようになる．
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label": label} # DataLoaderでbatch["image"]としてアクセス可能．
        return sample


# 自作を使用するなら
# custom_dataset = CustomImageDataset("labels.csv", "img_dir", transform=ToTensor())
# custom_dataloader = DataLoader(custom_dataset, batch_size=64, shuffle=True)

# DataLoaderの使用
from torch.utils.data import DataLoader
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# DataLoaderを用いた繰り返し処理
train_features, train_labels = next(iter(train_dataloader)) # イテレータ：順番に回せるオブジェクトの正体．next：最初の1バッチを取り出す．
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
print(f"Label: {label}")
os.makedirs("img_fashion", exist_ok=True)
plt.savefig("img_fashion/fashion_mnist_dataloader.png")
plt.close()