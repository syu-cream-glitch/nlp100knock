inputFile = "popular-names.txt"
firstColList = []
with open(inputFile, "r", encoding = "utf-8") as file:
    lines = file.readlines()
    for line in lines:
        firstCol = line.split("\t")[0]
        firstColList.append(firstCol)

firstColSet = set(firstColList)
print(firstColSet)
#実行確認$ cut -f1 2.UNIXコマンド/popular-names.txt | sort | uniq