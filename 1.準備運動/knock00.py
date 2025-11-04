word1 = 'パトカー'
word2 = 'タクシー'
mix_word = ''

word1_list = list(word1)
word2_list = list(word2)

for i in range(4):
  mix_word += word1_list[i] + word2_list[i]
print(mix_word, end = "")