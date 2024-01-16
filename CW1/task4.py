import numpy as np
from numpy.linalg import norm
import pandas as pd
from collections import Counter

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# to check iteration progress
from tqdm import tqdm


cand_passages = pd.read_csv('./candidate-passages-top1000.tsv', sep = '\t', header = None)
test_qry = pd.read_csv('./test-queries.tsv', sep = '\t', header = None)
idf_map = np.load('IDFMap_rm_stopwords.npy',allow_pickle='TRUE').item()
tf_map = np.load('TFMap_rm_stopwords.npy',allow_pickle='TRUE').item()

# filter duplicated passages, only extract passage
passages = cand_passages[~cand_passages.duplicated(subset=[1])]


lemm = WordNetLemmatizer()
def tokenization_nltk(line):
    line = word_tokenize(line)
    line = [lemm.lemmatize(word.lower()) for word in line if word.isalpha()]
    return line

N, _ = passages.shape
V = len(idf_map)
Ds = {pid:sum(tf_map[pid].values()) for pid in tf_map.keys()}
word_num = sum(Ds.values())


idf_keys_hash = dict()
for idx, k in enumerate(list(idf_map.keys())):
    idf_keys_hash[k] = idx
    
    
tf_keys_hash = dict()
for idx, k in enumerate(list(tf_map.keys())):
    tf_keys_hash[k] = idx

    # build query's vector
qs = np.zeros((test_qry.shape[0], V))

qry_counter = []

for qry_idx, q_row in tqdm(test_qry.iterrows()):
        
    c = Counter(tokenization_nltk(q_row[1]))
    qry_counter.append(c)


def compute_scores(socre_func):
    score_matrix = np.zeros((test_qry.shape[0] * 100, 3))
    idx = 0
    MAX_NUM_QRY = 100

    for qry_idx, q_row in tqdm(test_qry.iterrows()):

        qid = q_row[0]
        qry_passage = cand_passages[cand_passages[0] == qid]    
        qry_num = qry_passage.shape[0]
        max_qry_num = MAX_NUM_QRY if qry_num > MAX_NUM_QRY else qry_num

        scores, pids = np.zeros(qry_num), np.zeros(qry_num)

        qfs = []
        for qry_word in qry_counter[qry_idx].keys():
            if qry_word in idf_map: 
                qf = qry_counter[qry_idx][qry_word]
                qfs.append((qry_word, qf))

        for doc_idx in range(qry_num):
            pid = qry_passage[1].iloc[doc_idx]
            pids[doc_idx] = pid

            for qry_word, qf in qfs:
                scores[doc_idx] += socre_func(pid, qry_word)


        score_arg = np.argsort(scores)[::-1]
        score_arg = score_arg[:max_qry_num]

        update_range = np.arange(idx, idx + max_qry_num)
        score_matrix[update_range, 0] = qid
        score_matrix[update_range, 1] = pids[score_arg]
        score_matrix[update_range, 2] = scores[score_arg]

        idx += max_qry_num

    score_matrix = score_matrix[:idx]
    
    return score_matrix

# Laplace
def laplace_score_func(pid, qry_word):
    m = tf_map[pid][qry_word] if qry_word in tf_map[pid] else 0
    D = Ds[pid]
    
    return np.log((m + 1) / (D + V))

laplace_score = compute_scores(laplace_score_func)
np.savetxt("laplace.csv", laplace_score, delimiter=",")

# Lidstone
def lidstone_socre_func(pid, qry_word):
    eps = 0.1
    m = tf_map[pid][qry_word] if qry_word in tf_map[pid] else 0
    D = Ds[pid]
    
    return np.log((m + eps) / (D + eps * V))

lidstone_score = compute_scores(lidstone_socre_func)
np.savetxt("lidstone.csv", lidstone_score, delimiter=",")

# Dirichlet
def dirichlet_score_func(pid, qry_word):
    mu = 50
    m = tf_map[pid][qry_word] if qry_word in tf_map[pid] else 0
    D = Ds[pid]
    c = sum(idf_map[qry_word].values())

    return np.log((D / (D + mu)) * (m / D) + (mu / (D + mu)) * (c / word_num))

dirichlet_score = compute_scores(dirichlet_score_func)
np.savetxt("dirichlet.csv", dirichlet_score, delimiter=",")