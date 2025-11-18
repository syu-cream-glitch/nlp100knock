import os
import re
import collections

input_file = "output/knock20.txt"
os.makedirs("output", exist_ok = True)
output_file = os.path.join("output", "knock23.txt")

with open(input_file, "r", encoding = "utf-8") as input_f, \
    open(output_file, "w", encoding ="utf-8") as output_f:
    pattern = r"(={2,4}.*?={2,4})"
    lines = re.findall(pattern, input_f.read())
    for line in lines:
        freq = collections.Counter(re.findall(r"=", line))
        level = int(freq['='] / 2) - 1
        title = re.sub(r"=", "", line).strip()
        output_f.write(title + "\t" + str(level) + "\n")