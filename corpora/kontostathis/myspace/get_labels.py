"""Write myspace human judgement labels to one csv file."""

import csv
from glob import glob

fl = glob('./Human Concensus/*.csv')
hd = {}

for f in fl:
    reader = csv.reader(open(f))
    for row in reader:
        hd[row[0]] = 0 if row[1] == 'N' else 1

writer = csv.writer(open('labels.csv', 'w'))
for k, v in hd.items():
    writer.writerow([k, v])
