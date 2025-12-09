import os
from dotenv import load_dotenv
from gensim.models import KeyedVectors
import pandas as pd

load_dotenv()
path = os.getenv('W2V_MODEL_PATH')
wv = KeyedVectors.load_word2vec_format(path, binary=True)

input_file = 'data/questions-words.txt'
os.makedirs('output', exist_ok=True)
output_file = os.path.join('output', 'output54.csv')

def section_judge(text, keys):
    lines = text.splitlines()
    result_list = []
    section_flag = False

    for line in lines:
        if line.startswith(':'):
            section_flag = (line == f": {keys}")
            continue

        if section_flag:
            result_list.append(line)
    
    return result_list

records = []
with open(input_file, 'r', encoding='utf-8') as input_f:
    text = input_f.read()
    target_lines = section_judge(text, 'capital-common-countries')
    for line in target_lines[1:]:
        words = line.split()
        most_similar1 = wv.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=1)
        predicted_word, similarity = most_similar1[0]
        records.append({
            'word1': words[0],
            'word2': words[1],
            'word3': words[2],
            'answer': words[3],
            'predicted': predicted_word,
            'correct': predicted_word == words[3],
            'similarity': similarity
        })
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')





