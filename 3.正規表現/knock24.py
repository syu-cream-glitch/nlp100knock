import os
import re

input_file = "output/knock20.txt"
os.makedirs("output", exist_ok = True)
output_file = "output/knock24.txt"

with open(input_file, "r", encoding = "utf-8") as input_f, \
    open(output_file, "w", encoding = "utf-8") as output_f:
    pattern = r"\[\[ファイル:(.*?)(?:\||\])"
    lines = re.findall(pattern, input_f.read())
    for line in lines:
        output_f.write(line + "\n")
    