import torch

x = torch.ones(5)
y = torch.zeros(3)
w = torch.rand(5, 3, requires_grad=True) # requires_grad=True：逆伝播で勾配を計算する対象にする．
b = torch.rand(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
# 「logits（z）」を Sigmoid で確率に変換し，その確率と正解ラベル y との Binary Cross Entropy（BCE）を計算する

print('Gradient function for z =', z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)


# 逆伝播：勾配の計算
loss.backward()
print(w.grad)
print(b.grad)

# 勾配を計算しない
z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad(): # 勾配を追跡しないmode：メモリ節約と計算効率化，特に推論時に用いる
    z = torch.matmul(x, w) + b
print(z.requires_grad)

z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)
# 目的：ネットワークの一部のパラメータを固定したい．高速化したい．

# 順伝播の仕組み：指定された演算を実行し，計算結果のテンソルを求める．DAGの各操作gradient functionを更新
# 逆伝播の仕組み：loss.backward()が呼び出されると，各変数の.grad_fnを計算する，各変数の.grad属性に微分値を代入する，微分の連鎖律を使用して，各leafのテンソルの微分値を求める．

# テンソルに対する勾配とヤコビ行列
# PyTorchではbackwardを実行すると，勾配を蓄積する．
inp = torch.eye(5, requires_grad=True) # 単位行列5x5
out = (inp + 1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True) # retain_graph=True：同じ計算グラフをもう一度使えるように残す．
print("First call\n", inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print("Second call\n", inp.grad)
inp.grad.zero_() # 勾配を初期化しておく必要がある．（勾配の蓄積を避ける），今回は勾配の蓄積を確認するために準備
out.backward(torch.ones_like(inp), retain_graph=True)
print("After zeroing the gradients\n", inp.grad)