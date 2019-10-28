"""Slightly convoluted code for converting myspace xml into csv."""

import csv
import re
from glob import glob
from lxml import etree
from tqdm import tqdm

LABELS = open('labels.csv', 'r').read().split('\n')
LABELD = {}
for row in LABELS:
    if row:
        k, v = row.split(',')
        LABELD[k] = int(v)

ERRORS = []
WRITER = csv.writer(open('msp_set.csv', 'w'))


def clean_xml(xml_string, elem):
    """Escape special XML symbols like & < etc."""
    xml_string = str(xml_string)
    for body_string in re.findall(r'<{0}>.*<\/{0}>'.format(elem), xml_string):
        new_body = body_string.replace('<{0}>'.format(elem), '')
        new_body = new_body.replace('</{0}>'.format(elem), '')
        orig_copy = str(new_body)
        new_body = new_body.replace('<', ' &lt;')
        new_body = new_body.replace('<', ' &lt;')
        new_body = new_body.replace(']', '&#x005D;')
        xml_string = xml_string.replace(orig_copy, new_body)
    return xml_string


for i in tqdm(range(1, 12)):
    fl = glob('xml packet {0}'.format(i) + '/*.xml')
    for f in tqdm(fl, desc="folder {0}".format(i)):
        label = LABELD.get(f.split('/')[1].split('.xml')[0], 0)
        try:
            fs = open(f, 'r').read()
            if len(fs) > 20:
                fs = clean_xml(fs, 'body')
                fs = clean_xml(fs, 'username')
                fs = fs.replace('&', '&amp;')
                fs = fs.replace('encoding="UTF-8"', '')
                root = etree.fromstring(fs)
                # for some reason below cannot be in a one-liner
                line = []
                for x in root.findall('.//body'):
                    sentence = x.text
                    if sentence:
                        line.append(sentence)
                # ----------------------------------------------
                text = ' '.join(line)
                WRITER.writerow([label, text])
                # print("File {0} saved...".format(f))
        except etree.XMLSyntaxError:
            print("File {0} is malformed...".format(f))
            ERRORS.append(f)
        except TypeError:
            print("File {0} errored...".format(f))
            ERRORS.append(f)

print("Done!")
print("{0} errors found...".format(len(ERRORS)))
print("List: {0}".format(ERRORS))
