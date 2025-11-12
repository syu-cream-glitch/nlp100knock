import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
inputFile = "popular-names.txt"
start = time.time()
with open(inputFile, "r", encoding="utf-8") as file:
    linesCountSum = sum(1 for line in file)
end = time.time()
print(f"{linesCountSum}行でした．")
print(f"sum関数の処理時間:{end - start}秒")
print()

start = time.time()
with open("popular-names.txt", "r", encoding = "utf-8") as file:
    linesCountLen = len(file.readlines())
end = time.time()
print(f"{linesCountLen}行でした．")
print(f"readlinesでの処理時間:{end - start}秒")

#実行確認$ wc -l 2.UNIXコマンド/popular-names.txt
