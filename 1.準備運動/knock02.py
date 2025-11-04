word = 'stressed'
word_list = list(word)
for i in range(len(word_list), 0, -1):
  print(word_list[i - 1], end = "")