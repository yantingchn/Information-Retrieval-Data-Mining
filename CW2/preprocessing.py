import numpy as np
from numpy.linalg import norm
import pandas as pd
import re
from collections import Counter
# to check iteration progress
from tqdm import tqdm
tqdm.pandas()
import time

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import gensim
import gensim.downloader
from gensim.models import Word2Vec

import matplotlib.pyplot as plt



ignore_list = stopwords.words('english')
lemm = WordNetLemmatizer()
def tokenization_nltk(line):
    line = word_tokenize(line)
    line = [lemm.lemmatize(word.lower()) for word in line if (word.isalpha() and word not in ignore_list)]
    return line


################ Training Data ######################

train_data = pd.read_csv('./part2/train_data.tsv', sep = '\t')

tr_passages = train_data[~train_data.duplicated(subset=['pid'])].copy()
tr_passages['proc_passage'] = tr_passages['passage'].progress_apply(tokenization_nltk)

# build tf/idf map for training data
word_map = dict()

for idx, row in tqdm(tr_passages.iterrows()):
    for w in row['proc_passage']:
        if w not in word_map.keys():
            word_map[w] = 1
        else:
            word_map[w] += 1

print("Length of word map:", len(word_map.keys()))

# Build inverse document index map and terms' frequency map
idf_map = word_map
for key in idf_map.keys():
    idf_map[key] = dict()
    
tf_map = dict()

for index, row in tqdm(tr_passages.iterrows()):
    c = Counter(row['proc_passage'])
    pid = row['pid']
    tf_map[pid] = dict()
    for c_key in c.keys():
        if c_key in idf_map:
            idf_map[c_key][pid] = c[c_key]
            tf_map[pid][c_key] = c[c_key]

np.save('IDFMap_train.npy', idf_map)
np.save('TFMap_train.npy', tf_map)

################ Word2Vec ######################

st = time.time()
# Load training data
tr_for_w2v = tr_passages['proc_passage']

modelCBOW = gensim.models.Word2Vec(min_count=1, vector_size=100, window=3, sg=0, workers=16)
modelCBOW.build_vocab(tr_for_w2v, progress_per=10000)
modelCBOW.train(tr_for_w2v, total_examples=modelCBOW.corpus_count, epochs=5, report_delay=1)

print("EX Time:", time.time() - st)
modelCBOW.save("modelCBOW.model")


# set model
w2v_model = modelCBOW.wv

def get_passgae_vec(pid):
    vec = np.zeros(w2v_model.vector_size)
    w_count = 0
    for w in tf_map[pid].keys():
        if w in w2v_model:
            vec += (w2v_model[w] / norm(w2v_model[w])) * tf_map[pid][w]
            w_count += tf_map[pid][w]
        else:
            print ("not found:", w)
            
    if w_count > 0:
        return vec / w_count
    else:
        return vec
    
def get_val_passage_vec(p_list):
    vec = np.zeros(w2v_model.vector_size)
    w_count = 0
    for w in p_list:
        if w in w2v_model:
            vec += w2v_model[w] / norm(w2v_model[w])
            w_count += 1

    if w_count > 0:
        return vec / w_count
    else:
        return vec

def get_passage_length(pid):
    return sum(tf_map[pid].values()) / avdl


def get_qry_vec(qry):
    vec = np.zeros(w2v_model.vector_size)
    tok = tokenization_nltk(qry)    
    w_count = 0
    for w in tok:
        if w in w2v_model:
            vec += w2v_model[w] / norm(w2v_model[w])
            w_count += 1
    
    if w_count > 0:
        return vec / w_count
    else:
        return vec

def get_cosine_similarity(x):
    if x['q'].sum() == 0 or x['p'].sum() == 0:
        return 0
    else:
        return np.dot(x['p'], x['q']) / (norm(x['p']) * norm(x['q']))


def c_qd(x):
    tok = tokenization_nltk(x['queries'])
    count = 0
    for w in tok:
        if w in tf_map[x['pid']]:
            count += tf_map[x['pid']][w]
    
    return count


def q_idf(qry):
    tok = tokenization_nltk(qry)
    idfs = 0
    for w in tok:
        if w in idf_map:
            idfs += np.log10(N / (len(idf_map[w]) + 1))
    
    return idfs

def tfidf(x):
    qry = x['queries']
    pid = x['pid']
    
    tok = tokenization_nltk(qry)
    
    val = 0
    for w in tok:
        if w in tf_map[pid]:
            val += tf_map[pid][w] * np.log10(N / (len(idf_map[w]) + 1))
            
    return val

def l2dist(x):
    q = x['q']
    p = x['p']
    return norm(q - p)

def bm25(x):
    qry = x['queries']
    pid = x['pid']

    tok = tokenization_nltk(qry)
    c_tok = Counter(tok)
    val = 0
    
    for w in c_tok.keys():
        if w in tf_map[pid]:
            n = len(idf_map[w])
            f = idf_map[w][pid]
            qf = c_tok[w]
            K = k1 * ((1 - b) + b * (dl[pid] / avdl))

            val += np.log((N-n+0.5) / (n+0.5)) * (k1+1)*f / (K+f) * (k2+1)*qf / (k2+qf)
            
    return val

# negative sampling
posi_sample = train_data[train_data['relevancy'] == 1]
neg_sample = train_data[train_data['relevancy'] == 0].sample(600000)
new_tr = pd.concat([posi_sample, neg_sample]).sample(frac=1).reset_index(drop=True)
print("Positive sample:", posi_sample.shape[0], "Negative sample:", neg_sample.shape[0])

# load tf idf map
# idf_map = np.load('IDFMap_train.npy',allow_pickle='TRUE').item()
# tf_map = np.load('TFMap_train.npy',allow_pickle='TRUE').item()

N = train_data[~train_data.duplicated(subset=['pid'])].shape[0]

k1 = 1.2
k2 = 100
b = 0.75
dl = {pid:sum(tf_map[pid].values()) for pid in tf_map.keys()}
avdl = sum(dl.values())/len(dl)


new_tr['p'] = new_tr['pid'].progress_apply(get_passgae_vec)
new_tr['q'] = new_tr['queries'].progress_apply(get_qry_vec)
new_tr['doc_len'] = new_tr['pid'].progress_apply(get_passage_length)
new_tr['cosine'] = new_tr.progress_apply(get_cosine_similarity, axis=1)
new_tr['c_qd'] = new_tr.progress_apply(c_qd, axis=1)
new_tr['q_idf'] = new_tr['queries'].progress_apply(q_idf)
new_tr['tfidf'] = new_tr.progress_apply(tfidf, axis=1)
new_tr['l2dist'] = new_tr.progress_apply(l2dist, axis=1)
new_tr['bm25'] = new_tr.progress_apply(bm25, axis=1)

new_tr.to_pickle("task_tr.pkl")


################ Validation Data ######################

val_data = pd.read_csv('./part2/validation_data.tsv', sep = '\t')
val_passages = val_data[~val_data.duplicated(subset=['pid'])]
val_passages['proc_passage'] = val_passages['passage'].progress_apply(tokenization_nltk)


# build tf/idf map for training data
word_map = dict()

for idx, row in tqdm(val_passages.iterrows()):
    for w in row['proc_passage']:
        if w not in word_map.keys():
            word_map[w] = 1
        else:
            word_map[w] += 1

print("Length of word map:", len(word_map.keys()))

# Build inverse document index map and terms' frequency map
idf_map = word_map
for key in idf_map.keys():
    idf_map[key] = dict()
    
tf_map = dict()

for index, row in tqdm(val_passages.iterrows()):
    c = Counter(row['proc_passage'])
    pid = row['pid']
    tf_map[pid] = dict()
    for c_key in c.keys():
        if c_key in idf_map:
            idf_map[c_key][pid] = c[c_key]
            tf_map[pid][c_key] = c[c_key]

np.save('IDFMap_val.npy', idf_map)
np.save('TFMap_val.npy', tf_map)


N = val_data[~val_data.duplicated(subset=['pid'])].shape[0]

k1 = 1.2
k2 = 100
b = 0.75
dl = {pid:sum(tf_map[pid].values()) for pid in tf_map.keys()}
avdl = sum(dl.values())/len(dl)

val_data['proc_passage'] = val_data['passage'].progress_apply(tokenization_nltk)
val_data['p'] = val_data['proc_passage'].progress_apply(get_val_passage_vec)
val_data['q'] = val_data['queries'].progress_apply(get_qry_vec)
val_data['cosine'] = val_data.progress_apply(get_cosine_similarity, axis=1)
val_data['doc_len'] = val_data['pid'].progress_apply(get_passage_length)
val_data['c_qd'] = val_data.progress_apply(c_qd, axis=1)
val_data['q_idf'] = val_data['queries'].progress_apply(q_idf)
val_data['tfidf'] = val_data.progress_apply(tfidf, axis=1)
val_data['l2dist'] = val_data.progress_apply(l2dist, axis=1)
val_data['bm25'] = val_data.progress_apply(bm25, axis=1)

val_data.to_pickle("task_val.pkl")