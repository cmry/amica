from glob import glob
import json
import csv

fl = glob('./tweets/*.json')

school_writer = csv.writer(open('./twitter/school_set.csv', 'w'))
main_writer = csv.writer(open('./twitter/main_set.csv', 'w'))

for f in fl:
    with open(f, 'r') as jsf:
        js = json.loads(jsf.read())
        fw = school_writer if 'school' in js['source_set'] else main_writer
        fw.writerow([js['label_bullying'], js['text']])
