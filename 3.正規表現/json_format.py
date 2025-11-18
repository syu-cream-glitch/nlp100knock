import json
import os

input_file = "jawiki-country.json"
os.makedirs("output", exist_ok=True)
output_file = os.path.join("output", "knock20.json")  # 拡張子もjsonに変更

empty_list = []

# JSONファイルを読み込む
with open(input_file, "r", encoding="utf-8") as input_f:
    lines = input_f.readlines()
    for line in lines:
        empty_list.append(json.loads(line))

# イギリスの記事だけ抽出してJSON形式で保存
uk_data = {}
for entry in empty_list:
    if entry['title'] == 'イギリス':
        uk_data = entry['text']
        break

with open(output_file, "w", encoding="utf-8") as output_f:
    json.dump(uk_data, output_f, ensure_ascii=False, indent=4)