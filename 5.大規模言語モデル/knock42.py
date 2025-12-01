import os
import pandas as pd

#csvの中身：問題,選択肢1,選択肢2,選択肢3,選択肢4,正解
columns = ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'answer']
df = pd.read_csv("data/college_computer_science.csv", header = None, names = columns)

def load_jmml():
    jmml_lit = []
    df = pd.read_csv("data/college_computer_science.csv", header = None)
    for 