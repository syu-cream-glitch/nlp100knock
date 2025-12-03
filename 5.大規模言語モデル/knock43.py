import os
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd
import random

columns = ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'answer']
df = pd.read_csv("data/college_computer_science.csv", header=None, names=columns)

def load_jmml():
    jmml_list =[]
    df = pd.read_csv("data/college_computer_science.csv", header=None, names=columns)
    for index, row in df.iterrows():
        question = row['question']
        choices = [row['choice1'], row['choice2'], row['choice3'], row['choice4']]
        answer = row['answer']
        jmml = {
            "question": question,
            "choices": choices,
            "answer": answer
        }
        jmml_list.append(jmml)
    return jmml_list

jmml_data = load_jmml()

load_dotenv()

api_key = os.getenv("GENAI_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-2.5-flash')

#全データだとかなりの時間がかかるため、一部データのみ使用
jmml_data = jmml_data[:10]

#ベースのプロンプト
def create_jmml_prompt(jmml):
    prompt = f"""
以下は大学のコンピュータサイエンスの試験問題です。各問題に対して、最も適切な選択肢を1つ選んでください。
問題: {jmml['question']}
選択肢:
A. {jmml['choices'][0]}
B. {jmml['choices'][1]}
C. {jmml['choices'][2]}
D. {jmml['choices'][3]}
正しい選択肢をAからDの中から1つだけ答えてください。
出力はA、B、C、Dのいずれかにしてください。また，それ以外の説明は不要です。求めるのはアルファベット1文字だけです。
"""
    return prompt

#プロンプトを変更
def create_jmml_change_prompt(jmml):
    prompt = f"""
以下は大学のコンピュータサイエンスの試験問題です。各問題に対して、最も適切な選択肢を1つ選んでください。
問題: {jmml['question']}
選択肢:
ア. {jmml['choices'][0]}
イ. {jmml['choices'][1]}
ウ. {jmml['choices'][2]}
エ. {jmml['choices'][3]}
正しい選択肢をアからエの中から1つだけ答えてください。
なお，ア、イ、ウ、エはそれぞれA、B、C、Dに対応しています。
出力はA、B、C、Dのいずれかにしてください。また，それ以外の説明は不要です。求めるのはアルファベット1文字だけです。
"""
    return prompt

#ベースの評価関数
def evaluate_jmml(model, jmml_data):
    correct_count = 0
    total_count = len(jmml_data)

    for jmml in jmml_data:
        prompt = create_jmml_prompt(jmml)
        response = model.generate_content(prompt)
        answer = response.text.strip()
        if answer == jmml['answer']:
            correct_count += 1

    accuracy = correct_count / total_count
    return accuracy

accuracy_base = evaluate_jmml(model, jmml_data)

#プロンプトを変えた場合を適用した評価関数
def evaluate_jmml_change_prompt(model, jmml_data):
    correct_count = 0
    total_count = len(jmml_data)

    for jmml in jmml_data:
        prompt = create_jmml_change_prompt(jmml)
        response = model.generate_content(prompt)
        answer = response.text.strip()
        if answer == jmml['answer']:
            correct_count += 1

    accuracy = correct_count / total_count
    return accuracy

accuracy_change_prompt = evaluate_jmml_change_prompt(model, jmml_data)

#temperturreを変更した評価関数(0.0)
def evaluate_jmml_change0_prompt(model, jmml_data):
    correct_count = 0
    total_count = len(jmml_data)

    for jmml in jmml_data:
        prompt = create_jmml_prompt(jmml)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.0)
        )
        answer = response.text.strip()
        if answer == jmml['answer']:
            correct_count += 1

    accuracy = correct_count / total_count
    return accuracy

accuracy_change0 = evaluate_jmml_change0_prompt(model, jmml_data)

#temperturreを変更した評価関数(1.0)
def evaluate_jmml_change1_prompt(model, jmml_data):
    correct_count = 0
    total_count = len(jmml_data)

    for jmml in jmml_data:
        prompt = create_jmml_prompt(jmml)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=1.0)
        )
        answer = response.text.strip()
        if answer == jmml['answer']:
            correct_count += 1

    accuracy = correct_count / total_count
    return accuracy

accuracy_change1 = evaluate_jmml_change1_prompt(model, jmml_data)

def evaluate_jmml_shuffle_prompt(model, jmml_data):
    correct_count = 0
    total_count = len(jmml_data)

    for jmml in jmml_data:
        choices = jmml['choices'][:]
        random.shuffle(choices)
        prompt = create_jmml_prompt(jmml)
        response = model.generate_content(prompt)
        answer = response.text.strip()
        if answer == jmml['answer']:
            correct_count += 1

    accuracy = correct_count / total_count
    return accuracy

accuracy_shuffle = evaluate_jmml_shuffle_prompt(model, jmml_data)

output_file = os.path.join("output", "output43.txt")
os.makedirs("output", exist_ok=True)
with open(output_file, "w", encoding="utf-8") as output_f:
    output_f.write(f"ベースプロンプトの正答率: {accuracy_base:.2%}\n")
    output_f.write(f"プロンプト変更後の正答率: {accuracy_change_prompt:.2%}\n")
    output_f.write(f"temperature=0.0の正答率: {accuracy_change0:.2%}\n")
    output_f.write(f"temperature=1.0の正答率: {accuracy_change1:.2%}\n")
    output_f.write(f"選択肢シャッフル後の正答率: {accuracy_shuffle:.2%}\n")

#あとがき：プロンプト変更後の正答率以外は，対照実験にするためにベースプロンプトを使用している．