import torch
import numpy as np

# データから直接テンソルに変換
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# NumPyarrayからテンソルに変換
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 他のテンソルから作成
x_ones = torch.ones_like(x_data) # x_dataの特性（プロパティ）を維持（あくまでプロパティを維持するだけで，値はすべて1になる）
print(f"Ones Tensor: \n{x_ones}\n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # x_dataのdatatypeを上書き更新
print(f"Random Tensor: \n{x_rand}\n")

# ランダム値や定数のテンソルの作成
shape = (2, 3, ) # shapeはテンソルの次元を示すタプル：最後のカンマは1次元テンソルを示すために必要，次元の追加が容易になる．
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n{rand_tensor}\n")
print(f"Ones Tensor: \n{ones_tensor}\n")
print(f"Zeros Tensor: \n{zeros_tensor}\n")

# テンソルの属性変数：テンソルは属性変数として，形状，データの型，保存されているデバイスを保持している．
tensor = torch.rand(3, 4,)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}\n")

# GPUが利用可能であれば，GPU上にテンソルを移動させる
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

tensor = torch.ones(4, 4)
print('First row:', tensor[0])
print('First column:', tensor[:, 0])
print('Last column:', tensor[..., -1]) # 「...」はエリプシスで残りのすべての次元を表す．三次元以上からが分かり易い．
tensor[:, 1] = 0
print(tensor)
print()

# テンソルの結合
t1 = torch.cat([tensor, tensor, tensor], dim=1) # dim=0は行方向，dim=1は列方向への結合を意味する
print(t1)
print()

'''stackとの違い
a = [[1,1,1],
     [1,1,1]]
b = [[0,0,0],
     [0,0,0]]

cat([a,b], dim=0) =
[[1,1,1],
 [1,1,1],
 [0,0,0],
 [0,0,0]]

stackは新しい次元が追加される
stack([a,b], dim=0) =
[
 [[1,1,1],
  [1,1,1]],
 [[0,0,0],
  [0,0,0]]
]
'''

# 算術演算
# 2つのテンソルの行列の掛け算．
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3) # 結果をプロパティを維持して値をランダムにしたy3に保存する．
print(y1)
print()

# 要素ごとの積
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z1)
print()

# 1要素のテンソル
agg = tensor.sum()
agg_item = agg.item() # Pythonの数値として取得
print(agg_item, type(agg_item))
print()

# インプレース操作：演算結果をオペランドに格納する演算をインプレースと呼ぶ．
# メソッドの最後，接尾辞として捜査名に「_」が付く．
# 例えば，x.copy_(y)，x.t_()，x.add_(1)など．xの内容そのものを更新する．
# テンソル限定
print(tensor, "\n")
tensor.add_(5)
print(tensor, "\n")

# テンソル→NumPy
# NumPyとの変換：CPU上のテンソルとNumPy arraysは同じメモリを共有することができ，相互変換が容易．
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}\n")

# この際，テンソルが変化するとNumPy側も変化する．
t.add_(1)
print(f"t: {t}")
print(f"n: {n}\n")

# Numpy→テンソル
n = np.ones(5)
t = torch.from_numpy(n)

# NumPy arraysの変化はテンソル側にも反映される．
np.add(n, 1, out=n) # NumPy版のインプレース操作
print(f"t: {t}")
print(f"n: {n}\n")