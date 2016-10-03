import sys
import os
import json
import collections

from lshash import LSHash
import numpy as np

from collections import Counter, OrderedDict
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import re
from datetime import datetime
import string
import itertools

reload(sys)
sys.setdefaultencoding('utf-8')

from gensim import matutils

stop_en = stopwords.words('english')
stop_en = stop_en + list(['rt'])
stop_sp = stopwords.words('spanish')


def stem_doc(doc):
    #stemmer
    st = PorterStemmer()    
    stemmed_doc = [st.stem(word) for word in doc.split(" ")]
    return " ".join(stemmed_doc) 

def remove_all_punct(doc):
    exclude = set(string.punctuation)
    exclude.add('\r')
    exclude.add('\n')
    doc = ''.join(ch for ch in doc if ch not in exclude)
    return doc

def remove_punct(doc):
    exclude = set(string.punctuation)
    exclude.add('\r')
    exclude.add('\n')
    exclude.remove('#')
    doc = ''.join(ch for ch in doc if ch not in exclude)
    return doc

def remove_mentions(doc):
    noment_doc = [i for i in doc.lower().split() if '@' not in i]
    return " ".join(noment_doc) 

def remove_stop_words(doc):
    global stop_en
    stopped_doc = [i for i in doc.lower().split() if (i not in stop_en) and (i not in stop_sp)]
    return " ".join(stopped_doc) 

def remove_urls(txt):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', txt)
    for url in urls:
        txt = txt.replace(url,'')
    return txt

# we do not need to use this function because I added 'rt' to the stop words list
def remove_rt_str(txt):
    return txt.lower().replace('rt ', '')
   
def get_tags(text):
    tags = [remove_all_punct(i.encode("ascii","replace").replace('#',' ')) for i in text.lower().split() if '#' in i]
    return ' '.join([x for x in tags if x])


""" Calculate cosine similarity of two sparse vectors. """
def sparse_cos_sim(sv1, sv2):
    mag_prod = sparse_magnitude(sv1) * sparse_magnitude(sv2)
    if mag_prod == 0:
        return 0
    return float(sparse_dot_product(sv1, sv2) / mag_prod)


""" Calculate dot product of two sparse vectors. """  
def sparse_dot_product(sv1, sv2):
    d1 = dict(sv1)
    d2 = dict(sv2)
    tot = 0
    for key in set(d1.keys()).intersection(set(d2.keys())):
        tot += d1[key] * d2[key]
#         print key, tot
    return tot

""" Calculate magnitude of a sparse vector. """
def sparse_magnitude(sv):
    return sum(v**2 for (a, v) in sv)**0.5

""" Calculate dot product of a sparse vector 'sv' against a dense vector 'dv'.
    The sparse vector format is described below. No bounds checking is done,
    so make sure it doesn't exceed the size of 'dv'. """
def mixed_dot_product(sv, dv):
    tot = 0
    for (idx, val) in sv:
        try:
            tot += val * dv[idx]
        except:pass

    return tot

