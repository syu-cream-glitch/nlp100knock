import json
import os

input_file = "jawiki-country.json"
os.makedirs("output", exist_ok = True)
output_file = os.path.join("output", "knock20.txt")

empty_list = []

with open(input_file, "r", encoding = "utf-8") as input_f:
    lines = input_f.readlines()
    for line in lines:
        empty_list.append(json.loads(line))

with open(output_file, "w", encoding = "utf-8") as output_f:
    for i in range(len(empty_list)):
        if empty_list[i]['title'] == 'イギリス':
            output_f.write(empty_list[i]['text'])            
    #for item in empty_list:
    #    if item['title'] == 'イギリス':
    #        output_f.write(item['text'])  でもいい

"""
chatGPTに修正してもらうと
import json
import os

input_file = "jawiki-country.json"
os.makedirs("output", exist_ok=True)
output_file = os.path.join("output", "knock20.txt")

with open(input_file, "r", encoding="utf-8") as f_in, \
     open(output_file, "w", encoding="utf-8") as f_out:

    for line in f_in:
        article = json.loads(line)
        if article.get("title") == "イギリス":
            f_out.write(article.get("text", ""))
            break
"""