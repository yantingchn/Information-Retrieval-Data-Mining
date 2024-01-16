import numpy as np
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
word_map = np.load('WordMapRmStopLemm.npy',allow_pickle='TRUE').item()
# filter duplicated passages, only extract passage
passages = cand_passages[~cand_passages.duplicated(subset=[1])]


lemm = WordNetLemmatizer()
def tokenization_nltk(line):
    line = word_tokenize(line)
#     lines[i]=[word.lower() for word in lines[i] if word.isalpha()]        
    line = [lemm.lemmatize(word.lower()) for word in line if word.isalpha()]
    return line


# Build inverse document index map and terms' frequency map
idf_map = word_map
for key in idf_map.keys():
    idf_map[key] = dict()
    
tf_map = dict()
for index, row in tqdm(passages.iterrows()):
    c = Counter(tokenization_nltk(row[3]))
    pid = row[1]
    tf_map[pid] = dict()
    for c_key in c.keys():
        if c_key in idf_map:
            idf_map[c_key][pid] = c[c_key]
            tf_map[pid][c_key] = c[c_key]

np.save('IDFMap_rm_stopwords.npy', idf_map)
np.save('TFMap_rm_stopwords.npy', tf_map)