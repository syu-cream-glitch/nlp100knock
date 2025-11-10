inputFile = "popular-names.txt"
N = int(input("行数を指定してください:"))
with open(inputFile, "r", encoding = "utf-8") as file:
    displayData = file.readlines()[-N:]
    for line in displayData:
        print(line.rstrip("\n"))

#実行確認$ tail -n 10 2.UNIXコマンド/popular-names.txt