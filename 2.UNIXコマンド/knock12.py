import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding = 'utf-8')
inputFile = "popular-names.txt"
N = int(input())
with open(inputFile, "r", encoding = "utf-8") as file:
    displayData = file.readlines()[-N:]
    for line in displayData:
        print(line.rstrip("\n"))

#実行確認$ tail -n 10 2.UNIXコマンド/popular-names.txt