import gzip
import shutil

with gzip.open("jawiki-country.json.gz", "rt", encoding = "utf-8")as input_file:
    with open("jawiki-country.json", "wt", encoding = "utf-8") as output_file:
        shutil.copyfileobj(input_file, output_file)