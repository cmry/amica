import tweepy
import time
import csv
import json
import glob
from tqdm import tqdm
from itertools import zip_longest
from tweepy.error import RateLimitError


def attach_api():
    consumer_key = ''  # NOTE: add own tokens
    consumer_secret = ''

    access_token = ''
    access_token_secret = ''

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    return tweepy.API(auth)


api = attach_api()
dset = 'data'

reader = csv.reader(open(dset + '.csv', 'r'))
head = reader.__next__()


def batcher(n, iterable, fillvalue=(None,)*len(head)):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


tweet_batches = list(batcher(100, reader))


def get_batch(batch, fl):
    try:
        idl, uid, trace, type_, form, tease, role, emo = zip(*batch)

        fls, idls = set([f.replace('.json', '').replace('./outp/', '')
                         for f in fl]), set(idl)
        idl = list(idls - fls)
        print(idl)
        verified = 0
        for item in api.statuses_lookup(idl):
            i = idl.index(str(item.id))
            with open('./outp/' + str(idl[i]) + '.json', 'w') as jsf:
                if not item:
                    jsf.write("failed")
                else:
                    js = item._json
                    js['label_bullying'] = trace[i]
                    js['label_user'] = uid[i]
                    js['source_set'] = 'Xu'
                    js['label_type'] = type_[i]
                    js['label_form'] = form[i]
                    js['label_tease'] = tease[i]
                    js['label_role'] = role[i]
                    js['label_emo'] = emo[i]
                    json.dump(js, jsf)
            verified += 1
        print("Retrieved {0} tweets from this batch...".format(verified))

    except RateLimitError:
        print("zzzz")
        time.sleep(60)
        get_batch(batch, fl)
    except TypeError:
        print(batch)
        exit()


fl = glob.glob('./outp/*.json')
for batch in tqdm(tweet_batches):
    get_batch(batch, fl)
