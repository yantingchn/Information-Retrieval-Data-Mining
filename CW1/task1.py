import numpy as np

import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


import matplotlib.pyplot as plt

from tqdm import tqdm


lemm = WordNetLemmatizer()
def tokenization_nltk(line):
    line = word_tokenize(line)   # tokenisation   
    line = [lemm.lemmatize(word.lower()) for word in line if word.isalpha()] # normalization & lemmatisation
    return line


with open("./passage-collection.txt") as f:
    lines = f.readlines()
    
    count = 0
    for i in tqdm(range(len(lines))):
        count += 1
        lines[i] = tokenization_nltk(lines[i])



ignore_list = stopwords.words('english')

WordMap = dict()
for line in lines:
    for w in line:
        if w not in WordMap.keys():
            WordMap[w] = 1
        else:
            WordMap[w] += 1

print("Identified index of terms:", len(WordMap.keys()))


total_cnt = sum(WordMap.values())
sorted_WordMap = {k: v for k, v in sorted(WordMap.items(), key=lambda item: item[1], reverse=True)}

cnts = np.array(list(sorted_WordMap.values())) / total_cnt

N = cnts.shape[0]
zipf_sum = sum([1/n for n in range(1,N+1)])
zipf = lambda x: (x)**(-1) / zipf_sum    
zipf_x = np.linspace(1,N,N)

plt.plot(zipf_x, cnts, label = "data")
plt.plot(zipf_x, zipf(zipf_x), label = "theory (Zipf’s law)")

plt.yscale('log')
plt.xscale('log')

plt.xlabel("Term frequency ranking (log)")
plt.ylabel("Term prob. of occurrence (log)")
plt.legend()

plt.savefig("zipf.pdf", format = 'pdf')

plt.close()

# remove stopwords
WordMapRmStop = WordMap.copy()
for w in ignore_list:
    WordMapRmStop.pop(w, None)

# save dictionary
np.save('WordMapRmStopLemm.npy', WordMapRmStop)
print("Identified index of terms wo stop words:", len(WordMapRmStop.keys()))

total_cnt_rm_stop = sum(WordMapRmStop.values())
sorted_WordMapRmStop = {k: v for k, v in sorted(WordMapRmStop.items(), key=lambda item: item[1], reverse=True)}

cnts_rm_stop = np.array(list(sorted_WordMapRmStop.values())) / total_cnt_rm_stop

N = cnts_rm_stop.shape[0]
zipf_sum = sum([1/n for n in range(1,N+1)])
zipf = lambda x: (x)**(-1) / zipf_sum
zipf_x = np.linspace(1,N,N)

plt.plot(zipf_x, cnts_rm_stop, label = "data")
plt.plot(zipf_x, zipf(zipf_x), label = "theory (Zipf’s law)")


plt.yscale('log')
plt.xscale('log')

plt.xlabel("Term frequency ranking (log)")
plt.ylabel("Term prob. of occurrence (log)")
plt.legend()

plt.savefig("zipf_wo_stopword.pdf", format = 'pdf')

plt.close()