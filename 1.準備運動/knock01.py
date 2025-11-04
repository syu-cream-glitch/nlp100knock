word1 = 'パトカー'
word2 = 'タクシー'
mix_word = ''

word1_list = list(word1)
word2_list = list(word2)

for i in range(4):
  mix_word += word1_list[i] + word2_list[i]

mix_word_list = list(mix_word)
for i in range(1, len(mix_word_list), 2):
  print(mix_word_list[i], end = "")