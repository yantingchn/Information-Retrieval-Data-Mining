import numpy as np
from numpy.linalg import norm
import pandas as pd
import re
from collections import Counter

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


import matplotlib.pyplot as plt

from tqdm import tqdm
tqdm.pandas()
import time

# Run preprocessing.py before executing scripts for task1/2/3/4


lemm = WordNetLemmatizer()
def tokenization_nltk(line):
    line = word_tokenize(line)
#     lines[i]=[word.lower() for word in lines[i] if word.isalpha()]        
    line = [lemm.lemmatize(word.lower()) for word in line if word.isalpha()]
    return line
    
val_data = pd.read_csv('./part2/validation_data.tsv', sep = '\t')
passages = val_data[~val_data.duplicated(subset=['pid'])]
# passages['proc_passage'] = passages['passage'].progress_apply(tokenization_nltk)


# # build tf/idf map for training data
# word_map = dict()

# for idx, row in tqdm(passages.iterrows()):
#     for w in row['proc_passage']:
#         if w not in word_map.keys():
#             word_map[w] = 1
#         else:
#             word_map[w] += 1

# print("Length of word map:", len(word_map.keys()))

# # Build inverse document index map and terms' frequency map
# idf_map = word_map
# for key in idf_map.keys():
#     idf_map[key] = dict()
    
# tf_map = dict()

# for index, row in tqdm(passages.iterrows()):
#     c = Counter(row['proc_passage'])
#     pid = row['pid']
#     tf_map[pid] = dict()
#     for c_key in c.keys():
#         if c_key in idf_map:
#             idf_map[c_key][pid] = c[c_key]
#             tf_map[pid][c_key] = c[c_key]


# np.save('IDFMap_val.npy', idf_map)
# np.save('TFMap_val.npy', tf_map)

idf_map = np.load('IDFMap_val.npy',allow_pickle='TRUE').item()
tf_map = np.load('TFMap_val.npy',allow_pickle='TRUE').item()

N, _ = passages.shape
V = len(idf_map)
R = passages['relevancy'].sum().astype(int)

print(N, V, R)

idf_keys_hash = dict()
for idx, k in enumerate(list(idf_map.keys())):
    idf_keys_hash[k] = idx
    
    
tf_keys_hash = dict()
for idx, k in enumerate(list(tf_map.keys())):
    tf_keys_hash[k] = idx

    
# idf = lambda w : idf_map[w]
compute_idf = lambda word: np.log10(N / (len(idf_map[word]) + 1))


test_qry = val_data.drop_duplicates(subset=['qid'])[['qid', 'queries']]
test_qry = test_qry.reset_index(drop=True)
test_qry.head()


# build query's vector
qry_counter = []
for qry_idx, q_row in tqdm(test_qry.iterrows()):
    c = Counter(tokenization_nltk(q_row[1]))
    qry_counter.append(c)


def generate_query_vector(qry):
    qv = np.zeros(V)
    
    for qry_word in qry.keys():
        if qry_word in idf_map:
            posi = idf_keys_hash[qry_word]
            tf = qry[qry_word]
            qv[posi] = tf * compute_idf(qry_word)
            
    return qv


def generate_document_vector(pid):
    dv = np.zeros(V)    
    for word in tf_map[pid].keys():
        posi = idf_keys_hash[word]
        tf = tf_map[pid][word]
        dv[posi] = tf * compute_idf(word)

    return dv

# BM25 parameters
k1 = 1.2
k2 = 100
b = 0.75

dl = {pid:sum(tf_map[pid].values()) for pid in tf_map.keys()}
avdl = sum(dl.values())/len(dl)


def bm25_no_rel_score_func(pid, qry_word, qf):

    n = len(idf_map[qry_word])
    f = idf_map[qry_word][pid]
    K = k1 * ((1 - b) + b * (dl[pid] / avdl))

    return np.log((N-n+0.5) / (n+0.5)) * (k1+1)*f / (K+f) * (k2+1)*qf / (k2+qf)

def compute_scores(socre_func, data):
    score_matrix = np.zeros((test_qry.shape[0] * 1000, 4))
    idx = 0
    MAX_NUM_QRY = 1000

    for qry_idx, q_row in tqdm(test_qry.iterrows()):        
        qid = q_row[0]
        qry_passage = data[data['qid'] == qid]    
        qry_num = qry_passage.shape[0]
        max_qry_num = MAX_NUM_QRY if qry_num > MAX_NUM_QRY else qry_num

        scores, pids, rels = np.zeros(qry_num), np.zeros(qry_num), np.zeros(qry_num)
        
        qfs = []
        for qry_word in qry_counter[qry_idx].keys():
            if qry_word in idf_map: 
                qf = qry_counter[qry_idx][qry_word]
                qfs.append((qry_word, qf))

        for doc_idx in range(qry_num):
            pid = qry_passage['pid'].iloc[doc_idx]
            pids[doc_idx] = pid
            rels[doc_idx] = qry_passage['relevancy'].iloc[doc_idx]

            for qry_word, qf in qfs:
                if pid in idf_map[qry_word]:
                    scores[doc_idx] += socre_func(pid, qry_word, qf)

        score_arg = np.argsort(scores)[::-1]
        score_arg = score_arg[:max_qry_num]

        update_range = np.arange(idx, idx + max_qry_num)
        score_matrix[update_range, 0] = qid
        score_matrix[update_range, 1] = pids[score_arg]
        score_matrix[update_range, 2] = scores[score_arg]
        score_matrix[update_range, 3] = rels[score_arg]

        idx += max_qry_num

    score_matrix = score_matrix[:idx]
    return score_matrix

bm25_no_rel_score = compute_scores(bm25_no_rel_score_func, val_data)

# Average Precision
def avgACC(posi, top_num):
    r = posi.shape[0]
    prec = np.arange(r) + 1.0
    posi = posi[posi < top_num]
    avg_score = sum([prec[i] / (p+1) for i, p in enumerate(posi)]) / r

    return avg_score

def NDCG(posi, top_num):
    r = posi.shape[0]    
    posi = posi[posi < top_num]
    score = (1.0 / np.log2(posi + 2.0)).sum()
    
    optNDCG = (1.0 / np.log2(np.arange(r) + 2.0))[:top_num].sum()
    
    return score / optNDCG

def calc_accuracy(score_matrix, score_func, top_num=1000):
    MAX_EVAL_NUM = top_num

    accs = []
    qids = test_qry['qid'].unique()
    no_rel_count = 0
    for qid in tqdm(qids):
        qry_score = score_matrix[score_matrix[:,0] == qid]
        
        if qry_score.shape[0] < top_num:
            continue
        
        posi = np.where(qry_score[:, 3] == 1)[0]
        if posi.shape[0] == 0:
            accs.append(0)
        else:
            accs.append(score_func(posi, top_num))

    accs = np.array(accs)    
    return accs

bm25_no_rel_avg_acc = calc_accuracy(bm25_no_rel_score, avgACC, 100).mean()
bm25_no_rel_ndcg = calc_accuracy(bm25_no_rel_score, NDCG, 100).mean()

print("avg. Prec: {:.4f}".format(bm25_no_rel_avg_acc))
print("NDCG: {:.4f}".format(bm25_no_rel_ndcg))