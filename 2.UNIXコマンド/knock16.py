import random
import os

os.makedirs("output16", exist_ok = True)
inputFile = "popular-names.txt"
outputFile = os.path.join("output16", "knock16.txt")

with open(inputFile, "r", encoding = "utf-8") as file:
    lines = file.readlines()
    randomLines = random.sample(lines, len(lines))

with open(outputFile, "w", encoding = "utf-8") as outFile:
    outFile.writelines(randomLines)

#実行確認$ shuf 2.UNIXコマンド/popular-names.txt > output16/knock16.txt