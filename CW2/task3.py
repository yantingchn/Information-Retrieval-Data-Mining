import numpy as np
from numpy.linalg import norm
import pandas as pd
from collections import Counter

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

import xgboost as xgb


# Run preprocessing.py before executing scripts for task1/2/3/4

# load training data
tr = pd.read_pickle("task_tr.pkl")

# load validation data
val = pd.read_pickle("task_val.pkl")

test_qry = val.drop_duplicates(subset=['qid'])[['qid', 'queries']]
test_qry = test_qry.reset_index(drop=True)

### Feature selection 2 ###
selected_feature = ["cosine", 'q_idf', "c_qd", "tfidf", "l2dist", "bm25", 'q', 'p']
tr.sort_values(by=['qid'], inplace = True)
Xtr = np.hstack([np.vstack(tr[feat]) for feat in selected_feature])
ytr = np.array(tr['relevancy'], dtype = int)

Xval = np.hstack([np.vstack(val[feat]) for feat in selected_feature])
yval = np.array(val['relevancy'].copy(), dtype = int)

# Parameter grid 
params = {
        'learning_rate': [1e-4, 1e-3, 1e-2, 1e-1],
        'subsample': [0.5, 0.75, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [2, 4, 6, 8]
        }

# from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import make_scorer, ndcg_score

# ndcg_scorer = make_scorer(ndcg_score)

# skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1001)

# test_xgb = xgb.XGBRanker(  
#     tree_method='hist',
#     booster='gbtree',
#     objective='rank:ndcg',
#     random_state=41, 
#     eta=0.05, 
#     n_estimators=110, 
#     )

# random_search = RandomizedSearchCV(test_xgb, param_distributions=params, n_iter=5, scoring=ndcg_scorer, n_jobs=4, cv=skf.split(Xtr,ytr), verbose=3, random_state=41)
# random_search.fit(Xtr, ytr, qid = tr['qid'])

# best parameter setting
xgb_model = xgb.XGBRanker(  
    tree_method='hist',
    booster='gbtree',
    objective='rank:ndcg',
    eval_metric = 'ndcg-',
    random_state=41, 
    learning_rate=0.1,
    colsample_bytree=0.8, 
    eta=0.05, 
    max_depth=8, 
    n_estimators=110, 
    subsample=0.75 
    )


xgb_model.fit(Xtr, ytr, qid = tr['qid'], verbose=True)

val['pred'] = xgb_model.predict(Xval)

def lambdaMART_score(qry_passage, q_row):
    return qry_passage['pred'].to_numpy()


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

lambda_score = compute_scores(lambdaMART_score, val)

TOP_NUM = 100
lambda_avg_acc = calc_accuracy(lambda_score, avgACC, TOP_NUM).mean()
lambda_ndcg = calc_accuracy(lambda_score, NDCG, TOP_NUM).mean()

print("avg. Prec: {:.4f}".format(lambda_avg_acc))
print("NDCG: {:.4f}".format(lambda_ndcg))

# save results
with open('LM.txt', 'w') as f:
    qids = test_qry['qid'].unique()
    
    for qid in tqdm(qids):
        qry_score = lambda_score[lambda_score[:,0] == qid]
        
        if qry_score.shape[0] < TOP_NUM:
            continue

        for idx in range(TOP_NUM):
            pid = qry_score[idx, 1]
            score = qry_score[idx, 2]
            line = str(int(qid)) + " A2 " + str(int(pid)) + " " + str(idx+1) + " " + str(score) + " LM"
            f.write(line)
            f.write('\n')