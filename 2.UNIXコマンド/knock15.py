import os
import math

inputFile = "popular-names.txt"
os.makedirs("output", exist_ok = True)

N = int(input("分割数を指定してください:"))
with open(inputFile, "r", encoding = "utf-8") as file:
    lines = file.readlines()
    linesCount = len(lines)
    splitInformation  = math.ceil(linesCount / N)

for i in range(N):
    outputFile = os.path.join("output", f"output{i+1}.txt")
    with open(outputFile, "w", encoding = "utf-8") as outFile:
        outFile.writelines(lines[i * splitInformation : (i + 1) * splitInformation])

#実行確認$ split -l 10 2.UNIXコマンド/popular-names.txt output/unix_