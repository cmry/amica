import json

jsf = json.loads(open('simulated.json').read())
fo = open('simulated_conv.csv', 'w')
legal = ['Other_language', 'Harmless_sexual_talk', 'Good_characteristics']

for i, (k, v) in enumerate(jsf.items()):
    labs, sents = [], []
    for _k, _v in v['data'].items():
        try:
            labs += list(_v['labels'].keys())
        except IndexError:
            pass
        sents.append(_v['sentence'])

    min_lab = 0
    if labs and any([l not in legal for l in labs]):
        min_lab = 1

    sent_str = '\t'.join(sents)
    fo.write(f"{min_lab},{sent_str}" + "\n")

fo.close()
