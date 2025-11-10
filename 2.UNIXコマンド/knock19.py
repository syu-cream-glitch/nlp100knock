import os

os.makedirs("output19", exist_ok = True)
inputFile = "popular-names.txt"
outputFile = os.path.join("output19", "knock19.txt")

with open(inputFile, "r", encoding = "utf-8") as file:
    lines = file.readlines()

thirdColDesc = sorted(lines, key = lambda x: int(x.split("\t")[2]), reverse = True)

with open(outputFile, "w", encoding = "utf-8") as outFile:
    outFile.writelines(thirdColDesc)

#実行確認$ sort -k3,3nr 2.UNIXコマンド/popular-names.txt > output19/knock19.txt