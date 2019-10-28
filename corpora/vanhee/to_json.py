"""Converts the old AMiCA files to a more comprehensible json format."""

# pylint: disable=C0325,W0512,C0330

import csv
import json
from sys import argv
from glob import glob
from collections import OrderedDict


def annotations(folder):
    """Yield open annotation and text pairs."""
    files, err, load = set(), 0, 0
    for pair_part in glob(folder + '/*'):
        files.add('.'.join(pair_part.split('.')[:-1]))
    for floc in sorted(files):
        try:
            ann_f, txt_f = open(floc + '.ann', 'r'), open(floc + '.txt', 'r')
            yield floc, [x.split('\t') for x in ann_f.readlines()
                         ], txt_f.readlines()
            ann_f.close()
            txt_f.close()
            load += 1
        except FileNotFoundError:
            err += 1
    print("Found {0} errors while reading {1} files...".format(err, load))


def convert(ann, txt):
    """Convert .txt and .ann individually to an entry dictionary."""
    # get a index, sentence dictionary with trailing newlines removed
    text = {i: t for i, t in enumerate(
        [te.replace('\n', '') for te in txt if te != '\n'])}
    try:
        ann = {int(a[1].split()[1]): {
            "t": int(a[0][1:]),
            "index": tuple([int(x) for x in a[1].split()[1:]]),
            "text": None if '¶' in a[2] else a[2].replace('\n', ''),
            "label": a[1].split()[0]
            } for a in ann}
    except IndexError:
        print("Malformed file!")
        ann = {}
    return text, ann


def err_check(err, text, sentence):
    """Throw error if annotation does not seem to be in the original text."""
    # for some reason this yields some weird stuff
    print("\n\n\n\ntext:", text, "\nsnt:", sentence)
    # make a start_index annotation dictionary
    if err == "catch":
        try:
            assert text in sentence
        except AssertionError:
            pass
    else:
        assert text in sentence


def entry_to_data(entry, file_name, annotation, text, err=False):
    """Fill entry object with data extracted from .txt and .ann files."""
    pos = 0
    for i, sentence in OrderedDict(sorted(text.items())).items():
        labels, macro = {}, None
        for _ in sentence:
            if pos in annotation:
                ann = annotation[pos]
                if ann["index"][1] - ann["index"][0] == 1:
                    macro = ann["label"]
                else:
                    try:
                        labels[ann["label"]].append(ann["text"])
                    except KeyError:
                        labels[ann["label"]] = [ann["text"]]
                    if err:
                        err_check(err, ann["text"], sentence)
            pos += 1
        pos += 1
        # pos -= 4  # bug?  # nope?
        scope = ('q' if not i % 2 else 'a') if 'ask' in file_name else '?'
        entry["data"][i] = {
            "sentence": sentence.replace('¶ ', ''),
            "labels": labels,
            "macro": macro,
            "scope": scope
        }
    return entry


def files_to_dict(dataset_name):
    """Store the .txt and .ann files in a proper dictionary format."""
    data = {}
    for file_name, annotation, text in annotations(dataset_name):
        text, annotation = convert(annotation, text)
        entry = {"data": {}}
        entry = entry_to_data(entry, file_name, annotation, text)
        entry["annotations"] = annotation
        entry["text"] = text
        data[file_name] = entry
    return data


def dict_to_csv(data, name, flat=True):
    """Write dict format to .csv."""
    json.dump(data, open(name + '.json', 'w'), indent=4,
              separators=(',', ': '), sort_keys=True)
    writer = csv.writer(open(name + '.csv', 'w'), quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(["file_id", "scope", "label", "macro", "text"])
    i = 0  # don't remove
    for file_name, data_entry in data.items():
        for _, entry in data_entry["data"].items():
            label = ', '.join([x for x in entry["labels"].keys()] if not flat
                              else [x for x in entry["labels"].keys()][:1])
            label = 'Negative' if not label else label
            macro = 'Negative' if not entry["macro"] else entry["macro"]
            writer.writerow([file_name, entry["scope"], label, macro,
                            entry["sentence"]])


DATASET = argv[1]
DICT = files_to_dict(DATASET)
dict_to_csv(DICT, DATASET)
