import torch

#バージョン確認
print(torch.__version__)

#GPU利用可能か確認
print(torch.cuda.is_available())

#ランダムなテンソルを作成
x = torch.rand(5, 3)
print(x)