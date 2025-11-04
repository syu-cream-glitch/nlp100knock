import random
text = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
text_split = text.replace(".", "").split()

temp = ""
for txt in text_split:
    if len(txt) <= 4:
        pass
    else:
        txt = txt[0]+"".join(random.sample(txt[1:-1], len(txt[1:-1])))+txt[-1]
    temp += txt + " "
result = temp[:-1] + "."
print(result)
