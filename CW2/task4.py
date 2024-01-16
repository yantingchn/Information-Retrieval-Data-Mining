import numpy as np
from numpy.linalg import norm
import pandas as pd
from collections import Counter

import matplotlib.pyplot as plt

from tqdm import tqdm
tqdm.pandas()
import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

import multiprocessing
from datasets import load_dataset, Dataset

# Run preprocessing.py before executing scripts for task1/2/3/4

# load training data
tr = pd.read_pickle("task_tr.pkl")

# load validation data
val = pd.read_pickle("task_val.pkl")
test_qry = val.drop_duplicates(subset=['qid'])[['qid', 'queries']]
test_qry = test_qry.reset_index(drop=True)

### Feature selection 2 ###
selected_feature = ['q', 'p', "cosine", 'q_idf', "c_qd", "tfidf", "l2dist", "bm25"]

tr.sort_values(by=['qid'], inplace = True)

Xtr = np.hstack([np.vstack(tr[feat]) for feat in selected_feature])
ytr = tr['relevancy']

Xval = np.hstack([np.vstack(val[feat]) for feat in selected_feature])
yval = val['relevancy'].copy()


batch_size = 20480

# Combine the training inputs into a TensorDataset.
train_dataset = TensorDataset(torch.tensor(Xtr).float() , torch.tensor(ytr).float())
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )


# Combine the training inputs into a TensorDataset.
val_dataset = TensorDataset(torch.tensor(Xval).float() , torch.tensor(yval).float())
val_dataloader = DataLoader(
            val_dataset,  # The training samples.
            sampler = SequentialSampler(val_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

class LSTMSimilarity(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, feat_dim):
        super(LSTMSimilarity, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim*2 + feat_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        q = x[:, :100]
        p = x[:, 100:200]
        feat = x[:, 200:]
        lstm_output1, _ = self.lstm(q.view(len(q), 1, -1))
        lstm_output2, _ = self.lstm(p.view(len(p), 1, -1))
        x1_last = lstm_output1[:, -1, :]
        x2_last = lstm_output2[:, -1, :]
        x = torch.cat((x1_last, x2_last, feat), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        out = self.sig(x)
        return out.squeeze()
    

def train(data_loader, model, num_epochs=10, print_freq=1):
    
    optimizer = optim.Adam(model.parameters(), lr=1e-1)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(data_loader):

            optimizer.zero_grad()  # zero the gradients
            output = model.forward(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % print_freq == 0:
            print('epoch {} loss {}'.format(epoch+1, loss.item()))

    # classify = lambda q, p: model.forward(q, p)
    classify = lambda x: model.forward(x)
    return classify

model = LSTMSimilarity(100, 15, 6)
clf = train(train_dataloader, model, 20)

outputs = None
with torch.no_grad():
    for inputs, _ in tqdm(val_dataloader):
        # out = clf(inputs[:,:100], inputs[:,100:])
        out = clf(inputs)
        # print(f"Input: {inputs[0].item()}, Predicted Output: {output[0].item()}")
        # print(out.shape)
        # break
        if outputs is None:
            outputs = out
        else:
            outputs = torch.hstack([outputs, out])

    val['nn_score'] = outputs

def comp_nn_score(qry_passage, q_row):
    return qry_passage['nn_score'].to_numpy()

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

nn_score = compute_scores(comp_nn_score, val)

TOP_NUM = 100

nn_avg_acc = calc_accuracy(nn_score, avgACC, TOP_NUM).mean()
nn_ndcg = calc_accuracy(nn_score, NDCG, TOP_NUM).mean()

print("avg. Prec: {:.4f}".format(nn_avg_acc))
print("NDCG: {:.4f}".format(nn_ndcg))

# save results
with open('NN.txt', 'w') as f:
    qids = test_qry['qid'].unique()
    
    for qid in tqdm(qids):
        qry_score = nn_score[nn_score[:,0] == qid]
        
        if qry_score.shape[0] < TOP_NUM:
            continue

        for idx in range(TOP_NUM):
            pid = qry_score[idx, 1]
            score = qry_score[idx, 2]
            line = str(int(qid)) + " A2 " + str(int(pid)) + " " + str(idx+1) + " " + str(score) + " NN"
            f.write(line)
            f.write('\n')