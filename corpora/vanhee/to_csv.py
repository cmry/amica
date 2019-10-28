import csv
from sys import argv

reader = csv.reader(open(argv[1]))
writer = csv.writer(open(argv[2], 'w'))

for i, row in enumerate(reader):
    if not i:
        continue
    label, text = row[3], row[4]
    label = 0 if any([x in label for x in
                     ['Other_language', 'Negative']]) else 1
    writer.writerow([label, text])
