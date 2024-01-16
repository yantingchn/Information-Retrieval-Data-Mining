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

idf_keys_hash = dict()
for idx, k in enumerate(list(idf_map.keys())):
    idf_keys_hash[k] = idx
    
    
tf_keys_hash = dict()
for idx, k in enumerate(list(tf_map.keys())):
    tf_keys_hash[k] = idx
    
compute_idf = lambda word: np.log10(N / (len(idf_map[word]) + 1))

# build query's vector
qs = np.zeros((test_qry.shape[0], V))
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

#TF-IDF
print("TF-IDF")
tfidf_qry_score = np.zeros((test_qry.shape[0] * 100, 3))
tfidf_idx = 0
MAX_NUM_QRY = 100

for qry_idx, q_row in tqdm(test_qry.iterrows()):
        
    qid = q_row[0]
    qry_passage = cand_passages[cand_passages[0] == qid]
    qry_num = qry_passage.shape[0]
    cosine_score = np.zeros(qry_num)
    pids = np.zeros(qry_num)
    
    max_qry_num = MAX_NUM_QRY if qry_num > MAX_NUM_QRY else qry_num
    
    qv = generate_query_vector(qry_counter[qry_idx])
    
    for doc_idx in range(qry_num):
        pid = qry_passage[1].iloc[doc_idx]
        pids[doc_idx] = pid
        
        dv = generate_document_vector(pid)        
        cosine_score[doc_idx] = np.dot(dv, qv) / norm(dv)
    cosine_score /= norm(qv)
    
    score_arg = np.argsort(cosine_score)[::-1]
    score_arg = score_arg[:max_qry_num]
  

    update_range = np.arange(tfidf_idx, tfidf_idx + max_qry_num)
    tfidf_qry_score[update_range, 0] = qid
    tfidf_qry_score[update_range, 1] = pids[score_arg]
    tfidf_qry_score[update_range, 2] = cosine_score[score_arg]

    tfidf_idx += max_qry_num

tfidf_qry_score = tfidf_qry_score[:tfidf_idx]

np.savetxt("tfidf.csv", tfidf_qry_score, delimiter=",")



# BM25 parameters

k1 = 1.2
k2 = 100
b = 0.75

dl = []

# dl = {row[1]:len(tokenization_nltk(row[3])) for index, row in tqdm(passages.iterrows())}
dl = {pid:sum(tf_map[pid].values()) for pid in tf_map.keys()}
avdl = sum(dl.values())/len(dl)


# BM25
# no relevance information, R = r = 0
print("BM25")

bm25_score = np.zeros((test_qry.shape[0] * 100, 3))
bm25_idx = 0
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
            if pid in idf_map[qry_word]:
                n = len(idf_map[qry_word])
                f = idf_map[qry_word][pid]
                K = k1 * ((1 - b) + b * (dl[pid] / avdl))

                scores[doc_idx] += np.log((N-n+0.5) / (n+0.5)) * (k1+1)*f / (K+f) * (k2+1)*qf / (k2+qf)
            
    score_arg = np.argsort(scores)[::-1]
    score_arg = score_arg[:max_qry_num]


    update_range = np.arange(bm25_idx, bm25_idx + max_qry_num)
    bm25_score[update_range, 0] = qid
    bm25_score[update_range, 1] = pids[score_arg]
    bm25_score[update_range, 2] = scores[score_arg]

    bm25_idx += max_qry_num

bm25_score = bm25_score[:bm25_idx]

np.savetxt("bm25.csv", bm25_score, delimiter=",")