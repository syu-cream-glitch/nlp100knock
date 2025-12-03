import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

#csvの中身：問題,選択肢1,選択肢2,選択肢3,選択肢4,正解
columns = ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'answer']
df = pd.read_csv("data/college_computer_science.csv", header = None, names = columns)

def load_jmml():
    jmml_list = []
    df = pd.read_csv("data/college_computer_science.csv", header = None, names = columns)
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
genai.configure(api_key = api_key)

model = genai.GenerativeModel('gemini-2.5-flash')

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

accuracy = evaluate_jmml(model, jmml_data)

output_file = os.path.join("output", "output42.txt")
os.makedirs("output", exist_ok = True)
with open(output_file, "w", encoding="utf-8") as output_f:
    output_f.write(f"JMML Model Accuracy: {accuracy:.2%}\n")