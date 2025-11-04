def n_gram(target, n):
  return [ target[idx:idx + n] for idx in range(len(target) - n + 1)]

# 集合の作成
set_x = set(n_gram('paraparaparadise', 2))
print('X:', set_x)
set_y = set(n_gram('paragraph', 2))
print('Y:', set_y)

# 和集合
set_or = set_x | set_y
print('和集合:', set_or)

# 積集合
set_and = set_x & set_y
print('積集合:', set_and)

# 差集合
set_sub = set_x - set_y
print('差集合(X - Y):', set_sub)
set_sub = set_y - set_x
print('差集合(Y - X):', set_sub)

# 'se'が含まれるか？
print('seがXに含まれる:', 'se' in set_x)
print('seがYに含まれる:', 'se' in set_y)