import os
import re
import json

input_file = "output/knock20.txt"
os.makedirs("output", exist_ok = True)
output_file = os.path.join("output", "knock27.txt")

with open(input_file, "r", encoding = "utf-8") as input_f, \
    open(output_file, "w", encoding ="utf-8") as output_f:
    pattern = re.search(r"(?<=基礎情報)(.*)(?=\n\}\}\n)", input_f.read(), re.DOTALL).group()
    pattern = pattern.split('\n|')[1:] #国部分を削除
    info_dict = {}
    for line in pattern:
        key = re.search(r"(.*?)(?=\=)", line).group().replace(" ","") #replace(" ","")これは国 名のような場合のスペースも削除できる
        value = re.search(r"(?<=\=)(.*)", line).group().lstrip()
        value = re.sub("'''''", "", value)
        value = re.sub("'''", "", value)
        value = re.sub("''", "", value)
        value = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", value)
        info_dict[key.strip()] = value.strip()
    json.dump(info_dict, output_f, ensure_ascii=False, indent=4)