import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key = api_key)

model = genai.GenerativeModel("gemini-2.5-flash")

text = """
吾輩は猫である。名前はまだ無い。
どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。この書生というのは時々我々を捕えて煮て食うという話である。しかしその当時は何という考もなかったから別段恐しいとも思わなかった。ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。掌の上で少し落ちついて書生の顔を見たのがいわゆる人間というものの見始であろう。この時妙なものだと思った感じが今でも残っている。第一毛をもって装飾されべきはずの顔がつるつるしてまるで薬缶だ。その後猫にもだいぶ逢ったがこんな片輪には一度も出会わした事がない。のみならず顔の真中があまりに突起している。そうしてその穴の中から時々ぷうぷうと煙を吹く。どうも咽せぽくて実に弱った。これが人間の飲む煙草というものである事はようやくこの頃知った。
"""

prompt = f"""
次の文章のトークン数をトークンの定義に従って数えよ。
文章：{text}

トークンの定義：
    - 文章を処理する際の最小単位
    - 単語、句読点、特殊文字などを含む
    - スペースや改行もトークンとしてカウントする

出力形式：
トークン数: <数値>
実際の分割例: [<トークン1>, <トークン2>, ..., <トークンN>]
"""

response = model.generate_content(prompt)

#トークン数を計測する関数
token_count = model.count_tokens(text)

output_file = os.path.join("output", "output49.txt")
os.makedirs("output", exist_ok = True)
with open(output_file, "w", encoding = "utf-8") as output_f:
    output_f.write(response.text.strip() + "\n")
    output_f.write(f"\nモデルによるトークン数計測（プロンプトなし）: {token_count}\n")