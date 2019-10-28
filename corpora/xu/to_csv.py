import csv
import json
import re
from glob import glob

READER = csv.reader(open('data.csv'))
WRITER = csv.writer(open('./tweets/xu_set.csv', 'w'))
FILE_LIST = set(glob('./tweets/*.json'))

count = [0, 0]

for row in READER:
    tweet_id = './tweets/' + row[0] + '.json'
    label = 1 if any([x in row[6] for x in
                      ('bully', 'reinforcer', 'assistant')]) else 0
    label = 1 if 'cyberbullying' in row[3] + row[4] else 0 if not label else 1
    if tweet_id in FILE_LIST:
        js = json.load(open(tweet_id, 'r'))
        text = js['text']
        text = re.sub('cyber.*? |bull[y|i].*? ', '', text, flags=re.I)
        WRITER.writerow([label, text])
        count[label] += 1

print("Retrieved {0} negative and {1} positive...".format(*tuple(count)))
