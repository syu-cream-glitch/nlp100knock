inputFile = "popular-names.txt"
N = int(input("行数を指定してください:"))
with open(inputFile, "r", encoding = "utf-8") as file:
    displayData = file.readlines()[0:N]
    for line in displayData:
        firstCol = line.split("\t")[0]
        print(firstCol)

#実行確認$ head -n 10 2.UNIXコマンド/popular-names.txt | cut -f1