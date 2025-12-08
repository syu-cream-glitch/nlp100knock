import os
from dotenv import load_dotenv
from gensim.models import KeyedVectors

load_dotenv()
path = os.getenv('W2V_MODEL_PATH')
wv = KeyedVectors.load_word2vec_format(path, binary=True)

input_file = 'data/questions-words.txt'
os.makedirs('output', exist_ok=True)
output_file = os.path.join('output', 'output54.txt')

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


with open(input_file, 'r', encoding='utf-8') as input_f, \
    open(output_file, 'w', encoding='utf-8') as output_f:
    text = input_f.read()
    target_lines = section_judge(text, 'capital-common-countries')
    for line in target_lines[1:]:
        words = line.split()
        most_similar1 = wv.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=1)
        output_f.write(f"事例:{line}, 正解単語:{words[3]}, 求めた単語・類似度:{most_similar1[0][0]}・{most_similar1[0][1]}\n")




