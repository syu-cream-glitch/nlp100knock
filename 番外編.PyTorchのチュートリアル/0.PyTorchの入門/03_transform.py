from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    rott="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

# 大きさ10のゼロテンソルを作成し，sctter_を用いて，ラベルyの値のindexのみ1のワンホットエンコーディングに変換．