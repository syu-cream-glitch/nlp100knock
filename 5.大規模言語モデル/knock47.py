import os
import re
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key = api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

input_file = "output/output46.txt"
senryu_list = []
with open(input_file, "r", encoding = "utf-8") as input_f:
    content = input_f.read()
    pattern = r"(?:川柳：)\s*(.+)"
    senryu_list = re.findall(pattern, content)

senryu_evaluation_prompt = """
あなたは川柳の専門家です。
以下の各川柳について、面白さを10段階（1〜10）で評価し、その理由を簡潔に説明してください。

評価対象の川柳：
"""
for i, senryu in enumerate(senryu_list, start = 1):
    senryu_evaluation_prompt += f"{i}. {senryu}\n"

senryu_evaluation_prompt += """
出力形式は以下の通りにしてください。
1. 「川柳」
   評価:「1〜10の数値」
   理由:「簡潔な評価理由」
    
2. 「川柳」
   評価:「1〜10の数値」
   理由:「簡潔な評価理由」
    ...

10. 「川柳」
   評価:「1〜10の数値」
   理由:「簡潔な評価理由」
"""

response = model.generate_content(senryu_evaluation_prompt)

output_file = os.path.join("output", "output47.txt")
os.makedirs("output", exist_ok = True)
with open(output_file, "w", encoding = "utf-8") as output_f:
    output_f.write(response.text.strip())