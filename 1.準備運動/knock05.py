def n_gram(target, n):
  return [ target[idx:idx + n] for idx in range(len(target) - n + 1)]

print('bi-gram:', n_gram('I am an NLPer', 2))
print('tri-gram:', n_gram('I am an NLPer',3))