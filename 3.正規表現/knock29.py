import os
import requests
import json

input_file = "output/knock28.txt"
os.makedirs("output", exist_ok = True)
output_file = os.path.join("output", "knock29.txt")

with open(input_file, "r", encoding="utf-8") as input_f:
    data = json.loads(input_f.read())
S = requests.Session()
URL = "https://ja.wikipedia.org/w/api.php"
HEADERS = {
    # ユーザー固有のアプリケーション名と連絡先を設定してください。
    # 例: アプリ名/バージョン (メールアドレス)
    #実行時は自分のメールアドレスに置き換えた
    'User-Agent': 'nlp100knock_chapter3_29/1.0 (your.email@example.com)' 
}
S.headers.update(HEADERS)

PARAMS = {
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "titles": "ファイル:" + data['国旗画像'],
    "iiprop": "url"
}

R = S.get(url=URL, params=PARAMS)
DATA = R.json()
PAGES = DATA["query"]["pages"]
with open(output_file, "w", encoding="utf-8") as output_file:
    for key, value in PAGES.items():
        output_file.write(value["imageinfo"][0]["url"] + "\n")