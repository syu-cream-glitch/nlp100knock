import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding = 'utf-8')
import collections
inputFile = "popular-names.txt"
firstColList = []

with open(inputFile, "r", encoding = "utf-8") as file:
    lines = file.readlines()
    for line in lines:
        firstCol = line.split("\t")[0]
        firstColList.append(firstCol)

countFirstCol = collections.Counter(firstColList)
for name, count in countFirstCol.most_common():
    print(f"{count}\t{name}")

#実行確認$ cut -f1 2.UNIXコマンド/popular-names.txt | sort | uniq -c | sort -k1,1nr -k2,2