import collections
s = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
empty_list = []

s_split = s.replace(',','').replace('.','').split()
for i in range(len(s_split)):
  empty_list.append(len(s_split[i]))

count_list = collections.Counter(empty_list)
desc_list = list(count_list.keys())
print(desc_list)