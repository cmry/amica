import csv
import frog
from tqdm import tqdm

frogger = frog.Frog(frog.FrogOptions(parser=False, ner=False))
reader = csv.reader(open('dmad_a.csv'))
writer = csv.writer(open('dmad_a_tagged.csv', 'w'))

corp = [x for x in reader]

for i, r in enumerate(tqdm(corp)):
    try:
        r += ["\n".join(["\t".join([token["text"], token["lemma"], token["pos"]]) for token in frogger.process(r[4])])] if i else ['frogs']
        writer.writerow(r)
    except IndexError:
        print(r)
