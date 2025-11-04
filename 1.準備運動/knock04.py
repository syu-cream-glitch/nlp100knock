s = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
empty_list = []
s_split = s.replace(',','').replace('.','').split()

def wordTake(num , word):
    if num in [1,5,6,7,8,9,15,16,19]:
        return (word[0],num)
    else:
        return(word[:2],num)

for num, word in enumerate(s_split, 1):
    empty_list.append(wordTake(num, word))
print(dict(empty_list))