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
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import gensim
import gensim.downloader
from gensim.models import Word2Vec


import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt


# Run preprocessing.py before executing scripts for task1/2/3/4

ignore_list = stopwords.words('english')

lemm = WordNetLemmatizer()
def tokenization_nltk(line):
    line = word_tokenize(line)
    line = [lemm.lemmatize(word.lower()) for word in line if (word.isalpha() and word not in ignore_list)]
    return line


train_data = pd.read_csv('./part2/train_data.tsv', sep = '\t')

# tr_passages = train_data[~train_data.duplicated(subset=['pid'])].copy()
# tr_passages['proc_passage'] = tr_passages['passage'].progress_apply(tokenization_nltk)

# st = time.time()
# # Load training data
# tr_for_w2v = tr_passages['proc_passage']

# modelCBOW = gensim.models.Word2Vec(min_count=1, vector_size=100, window=3, sg=0, workers=16)
# modelCBOW.build_vocab(tr_for_w2v, progress_per=10000)
# modelCBOW.train(tr_for_w2v, total_examples=modelCBOW.corpus_count, epochs=5, report_delay=1)

# print("EX Time:", time.time() - st)
# modelCBOW.save("modelCBOW.model")
modelCBOW = Word2Vec.load("modelCBOW.model")


# # build tf/idf map for training data
# word_map = dict()

# for idx, row in tqdm(tr_passages.iterrows()):
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

# for index, row in tqdm(tr_passages.iterrows()):
#     c = Counter(row['proc_passage'])
#     pid = row['pid']
#     tf_map[pid] = dict()
#     for c_key in c.keys():
#         if c_key in idf_map:
#             idf_map[c_key][pid] = c[c_key]
#             tf_map[pid][c_key] = c[c_key]

# np.save('IDFMap_train.npy', idf_map)
# np.save('TFMap_train.npy', tf_map)


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


# # negative sampling
# posi_sample = train_data[train_data['relevancy'] == 1]
# neg_sample = train_data[train_data['relevancy'] == 0].sample(600000)
# new_tr = pd.concat([posi_sample, neg_sample]).sample(frac=1).reset_index(drop=True)
# print("Positive sample:", posi_sample.shape[0], "Negative sample:", neg_sample.shape[0])

# load tf idf map
idf_map = np.load('IDFMap_train.npy',allow_pickle='TRUE').item()
tf_map = np.load('TFMap_train.npy',allow_pickle='TRUE').item()

N = train_data[~train_data.duplicated(subset=['pid'])].shape[0]

k1 = 1.2
k2 = 100
b = 0.75
dl = {pid:sum(tf_map[pid].values()) for pid in tf_map.keys()}
avdl = sum(dl.values())/len(dl)


# new_tr['p'] = new_tr['pid'].progress_apply(get_passgae_vec)
# new_tr['q'] = new_tr['queries'].progress_apply(get_qry_vec)
# new_tr['doc_len'] = new_tr['pid'].progress_apply(get_passage_length)
# new_tr['cosine'] = new_tr.progress_apply(get_cosine_similarity, axis=1)
# new_tr['c_qd'] = new_tr.progress_apply(c_qd, axis=1)
# new_tr['q_idf'] = new_tr['queries'].progress_apply(q_idf)
# new_tr['tfidf'] = new_tr.progress_apply(tfidf, axis=1)
# new_tr['l2dist'] = new_tr.progress_apply(l2dist, axis=1)
# new_tr['bm25'] = new_tr.progress_apply(bm25, axis=1)

# new_tr.to_pickle("task_tr.pkl")

new_tr = pd.read_pickle("task_tr.pkl")


### Feature selection ###
p = torch.cat([torch.tensor(x)[None,:] for x in new_tr['p']])
q = torch.cat([torch.tensor(x)[None,:] for x in new_tr['q']])
X = torch.cat((
        torch.tensor(new_tr['cosine'].values) [None, :],
        torch.tensor(new_tr['q_idf'].values) [None, :],
        torch.tensor(new_tr['l2dist'].values) [None, :],
        torch.tensor(new_tr['bm25'].values) [None, :],
)).T
X = torch.hstack((X, q, p))

y = torch.tensor(new_tr['relevancy'].values)


# Logistic model
class LogisticClassifier(nn.Module):

    def __init__(self, dim):
        super(LogisticClassifier, self).__init__()
        self.w = nn.Parameter(torch.randn(dim, 1), requires_grad=True)
        self.b = nn.Parameter(torch.randn(1), requires_grad=True)
        
    def forward(self, x):
        return torch.sigmoid((x @ self.w + self.b)).squeeze()


def logistic_loss(y_pred, y_true): # define loss function of LR model
    loss = -torch.mean(y_true * torch.log(y_pred) + (1.0 - y_true) * torch.log(1.0 - y_pred))
    return loss


def LR_train(xTr, yTr, num_epochs=1000, lr=1e-3, print_freq=200):

    model = LogisticClassifier(dim=xTr.shape[1]).double()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        y_pred = model.forward(xTr)
        
        # loss = logistic_loss(y_pred, yTr) + (torch.norm(model.w) ** 2)
        loss = criterion(y_pred, yTr)
        loss.backward()
        optimizer.step()
    
        if (epoch + 1) % print_freq == 0:
            print('epoch {} loss {}'.format(epoch+1, loss.item()))

    # LRclassify = lambda x: model.forward(x).detach()
    return model, loss.item()

LR_model, loss = LR_train(X, y, num_epochs=1000, lr=0.2)
print("Model Traing Finished")

lrs = 5.0 ** np.linspace(-5, 3, 9)
losses = []
for lr in lrs:
    _, loss = LR_train(X, y, num_epochs=1000, lr=lr, print_freq=2000)
    losses.append(loss)
    print(lr, loss)

plt.semilogx(lrs, losses, 'o-')
for (x,y) in zip(lrs, losses):                                       
    plt.annotate('(%.4f)' % y, xy=(x, y), xytext=(x, y + 0.015)) 
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
# plt.savefig('lr_vs_loss.png')



# validation
val_data = pd.read_csv('./part2/validation_data.tsv', sep = '\t')
val_data = val_data.sort_values(by=['qid'])

test_qry = val_data.drop_duplicates(subset=['qid'])[['qid', 'queries']]
test_qry = test_qry.reset_index(drop=True)


# load tf idf map
idf_map = np.load('IDFMap_val.npy',allow_pickle='TRUE').item()
tf_map = np.load('TFMap_val.npy',allow_pickle='TRUE').item()

N = val_data[~val_data.duplicated(subset=['pid'])].shape[0]

k1 = 1.2
k2 = 100
b = 0.75
dl = {pid:sum(tf_map[pid].values()) for pid in tf_map.keys()}
avdl = sum(dl.values())/len(dl)

# val_data['proc_passage'] = val_data['passage'].progress_apply(tokenization_nltk)
# val_data['p'] = val_data['proc_passage'].progress_apply(get_val_passage_vec)
# val_data['q'] = val_data['queries'].progress_apply(get_qry_vec)
# val_data['cosine'] = val_data.progress_apply(get_cosine_similarity, axis=1)
# val_data['doc_len'] = val_data['pid'].progress_apply(get_passage_length)
# val_data['c_qd'] = val_data.progress_apply(c_qd, axis=1)
# val_data['q_idf'] = val_data['queries'].progress_apply(q_idf)
# val_data['tfidf'] = val_data.progress_apply(tfidf, axis=1)
# val_data['l2dist'] = val_data.progress_apply(l2dist, axis=1)
# val_data['bm25'] = val_data.progress_apply(bm25, axis=1)

# val_data.to_pickle("task_val.pkl")  
val_data = pd.read_pickle("task_val.pkl")  

def LR_predict_proba(X):
    pred = torch.zeros(X.shape[0])
    with torch.no_grad():
        for i in range(X.shape[0]):
            pred[i] = LR_model.forward(X[i,:])
    
    return pred 


# Q-D vectors
# cosine sim / doc_len / c_qd / q_idf
def lr_score_v2(qry_passage, q_row):
    tmp_p = torch.cat([torch.tensor(x)[None,:] for x in qry_passage['p']])
    tmp_q = torch.cat([torch.tensor(x)[None,:] for x in qry_passage['q']])

    tmp_x = torch.cat((
                torch.tensor(qry_passage['cosine'].values)[None, :], 
                torch.tensor(qry_passage['q_idf'].values)[None, :], 
                torch.tensor(qry_passage['l2dist'].values)[None, :],
                torch.tensor(qry_passage['bm25'].values)[None, :],
    )).T

    tmp_x = torch.hstack((tmp_x, tmp_q, tmp_p))

    # scores = LR_model.predict_proba(tmp_x)[:, 1]
    scores =  LR_predict_proba(tmp_x) 
    return np.array(scores)


def compute_scores(score_func, data):
    score_matrix = np.zeros((test_qry.shape[0] * 1000, 4))
    idx = 0
    MAX_NUM_QRY = 1000

    for qry_idx, q_row in tqdm(test_qry.iterrows()):        
        qid = q_row[0]
        qry_passage = data[data['qid'] == qid]    
        qry_num = qry_passage.shape[0]
        max_qry_num = MAX_NUM_QRY if qry_num > MAX_NUM_QRY else qry_num

        scores = np.zeros(qry_num)
        pids, rels = qry_passage['pid'].to_numpy(), qry_passage['relevancy'].to_numpy()     

        scores = score_func(qry_passage, q_row)
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
            no_rel_count += 1
            accs.append(0)
        else:
            accs.append(score_func(posi, top_num))

    accs = np.array(accs)
    return accs


lr_scores = compute_scores(lr_score_v2, val_data)
assert lr_scores[0:1000, 3].max() == 1.0

TOP_NUM = 100
lr_avg_acc = calc_accuracy(lr_scores, avgACC, TOP_NUM).mean()
lr_ndcg = calc_accuracy(lr_scores, NDCG, TOP_NUM).mean()

print("avg. Prec: {:.4f}".format(lr_avg_acc))
print("NDCG: {:.4f}".format(lr_ndcg))

# save results
with open('LR.txt', 'w') as f:
    qids = test_qry['qid'].unique()
    
    for qid in tqdm(qids):
        qry_score = lr_score_v2[lr_score_v2[:,0] == qid]
        
        if qry_score.shape[0] < TOP_NUM:
            continue

        for idx in range(TOP_NUM):
            pid = qry_score[idx, 1]
            score = qry_score[idx, 2]
            line = str(int(qid)) + " A2 " + str(int(pid)) + " " + str(idx+1) + " " + str(score) + " LR"
            f.write(line)
            f.write('\n')