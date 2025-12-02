import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key = api_key)

model = genai.GenerativeModel('gemini-2.5-flash')

prompt = """
つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。

以下の情報を参考にして，最終的な解答は「解答:〇〇駅」のように記載すること。また，解答以外は記載しないこと。
東急東横線: 渋谷 → 代官山 → 中目黒 → 祐天寺 → 学芸大学 → 都立大学 → 自由が丘 → 田園調布 → 多摩川 → 新丸子 → 武蔵小杉...

東急大井町線: 大井町 → 下神明 → 戸越公園 → 中延 → 荏原町 → 旗の台 → 北千束 → 大岡山 → 緑が丘 → 自由が丘 → 九品仏 → 尾山台 → 等々力 → 上野毛 → 二子玉川 → 二子新地 → 高津 → 溝の口 → 梶が谷 → 宮崎台 → 宮前平 → 鷺沼 → たまプラーザ → あざみ野 → 江田 → 市が尾 → 藤が丘 → 青葉台 → 田奈 → 長津田 → つきみ野 → 中央林間

東急大井町線の急行停車駅: 大井町、大岡山、自由が丘、二子玉川、溝の口、長津田、中央林間
"""

response = model.generate_content(prompt)

output_file = os.path.join("output", "output44.txt")
os.makedirs("output", exist_ok = True)
with open(output_file, "w", encoding = "utf-8") as output_f:
    output_f.write(response.text.strip())