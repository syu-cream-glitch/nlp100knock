import os
import pandas as pd

input_file = 'output/output54.csv'
os.makedirs('output', exist_ok=True)
output_file = os.path.join('output', 'output55.txt')

df = pd.read_csv(input_file)
#semantic_analogy_accuracyの計算
correct_count = df['correct'].sum()
total_count = len(df)
semantic_analogy_accuracy = correct_count / total_count

os.makedirs('output', exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as output_f:
    output_f.write(f"semantic analogy accuracy: {correct_count}/{total_count} = {semantic_analogy_accuracy}\n")
