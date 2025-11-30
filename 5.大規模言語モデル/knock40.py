import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key = api_key)

model = genai.GenerativeModel('gemini-2.5-flash')

prompt = """
9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。なお、解答は「解答: ア→イ→ウ」のように記載すること。
また，解答以外は記載しないこと。

ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。
イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。
解答:
"""

response = model.generate_content(prompt)

output_file = os.path.join("output", "output40.txt")
os.makedirs("output", exist_ok = True)
with open(output_file, "w", encoding = "utf-8") as output_f:

    output_f.write(response.text.strip())