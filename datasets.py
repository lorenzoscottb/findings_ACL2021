
import os
import spacy
import argparse
import time
import pickle
import torch
import torch.utils.data
import numpy as np
from numba import jit, njit
from tqdm import tqdm

class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, inputFileName, min_count, ratio=1.0, pwr=0.75, fl_type='single_str'):
        self.negatives = []
        self.discards = []
        self.negpos = 0
        self.power = pwr
        self.m_unigram = None
        self.word2id = dict()
        self.id2word = dict()
        self.token_count = 0
        self.word_frequency = dict()
        self.ratio = ratio
        self.inputFileName = inputFileName
        self.read_words(min_count, ratio, fl_type)
        self.initTableNegatives()
        self.initTableDiscards()

    def read_words(self, min_count, ratio, fl_type):
        print("Reading data")
        word_frequency = dict()
        if fl_type == 'single_str':
            word_sequence = open(self.inputFileName, encoding="utf8").read().replace("\n", " ").split()
        else:
            word_sequence = open(self.inputFileName, 'r')
            word_sequence = word_sequence.readlines()
            word_sequence = [word for s in tqdm(word_sequence) for word in s.split() if word != '"']
        word_sequence = word_sequence[:int(ratio*len(word_sequence))]
        print('Collecting tokens')
        for word in tqdm(word_sequence):
            if len(word) > 0:
                self.token_count += 1
                word_frequency[word] = word_frequency.get(word, 0) + 1
        print("\nTotal tokens: " + str(self.token_count))

        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
            print("Word-Vocabulary size: " + str(len(self.word2id)), end='\r')
        print("Vocabulary size: " + str(len(self.word2id)))
        
    def compute_mikolv_dist(self): 
        print('compute Mikolv distribution')   
        elevated = np.power(np.array(list(self.word_frequency.values()), dtype='int'), 
                            self.power)
        s_e = sum(elevated)
        self.m_unigram = {i:elevated[c]/s_e for c, (i, v) in enumerate(self.word_frequency.items())}

    def compute_unk_token(self):
        print ('compute unk') 
        for k in list(self.token_count.keys()):
            if self.token_count[k] < self.min_count and k != '_UNK':
                self.token_count['_UNK'] = self.token_count.get('_UNK')+self.token_count.get(k)
                del self.token_count[k]     
                
    def initTableDiscards(self):
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.5
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def getNegatives(self, target, size):
        response = self.negatives[self.negpos:self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(response) != size:
            return np.concatenate((response, self.negatives[0:self.negpos]))
        return response
    
    def compute_neg_sample_tensor(self, neg_dict_len=100000000):
        if self.m_unigram is None:
            self.compute_mikolv_dist()
        print('Collecting negative samples array...')
        idx, dst = zip(*list(self.m_unigram.items()))
        self.neg_sample_array = np.random.choice(idx, # words(id)
                                                neg_dict_len, # how many smpls at time
                                                p=dst) # gets probability if word(id)
        


class DMDataset(torch.utils.data.Dataset):
    def __init__(self, data, neg_num, fl_type='single_str', load_data=None, parser='spacy', use_gpu=False):
        self.data = data
        self.ns = neg_num
        print('Preparing DMSGNS dataset...')
        if load_data is not None:
            sk = self.load_file(load_data)
        else:
            sk = self.collect_sk_dep(fl_type, parser=parser, use_gpu=use_gpu)
        st = set([v[1] for v in sk])
        self.dep2id = {w:c for c,w in enumerate(st)}
        self.id2dep = {c:w for (w,c) in self.dep2id.items()}
        self.id_sk = [(self.data.word2id[w[0]], self.dep2id[w[1]] , self.data.word2id[w[2]])
                      for w in sk if  w[0] in self.data.word2id.keys() 
                      and w[2] in self.data.word2id.keys()]
        self.data_len = len(self.id_sk)
        print("Dependency-Vocabulary size: " + str(len(self.dep2id)))
        print("Dataset size: "+str(self.data_len))

    def load_file(self, dir_data):
        print('Loading dmsk data...')
        filehandler = open(dir_data, 'rb') 
        sk = pickle.load(filehandler)
        return sk
        
    def collect_sk_dep(self, fl_type, parser='spacy', use_gpu=False):
        spc_mx = 1000000
        strt = 0
        if fl_type == 'single_str':
            words = open(self.data.inputFileName, encoding="utf8").read().replace("\n", " ").split()
        else:
            words = open(inputFileName, 'r')
            words = words.readlines()
            words = [word for s in tqdm(words) for word in s.split() if word != '"']
        words = words[:int(self.data.ratio*len(words))]
        self.word_ids = [self.data.word2id[w] for w in words if
                         w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]
        self.data_len = len(self.word_ids) 
        if parser == 'spacy':       
            nlp = spacy.load("en_core_web_lrg")
            sk = [] 
            for r in tqdm(range(int(len(words) / spc_mx))):
                try:
                    end = spc_mx*(r+1)-1
                    doc = nlp(words[strt:end])
                    for word in doc:
                        if word.dep_ != 'ROOT':
                            sk.append((word.text,
                                        '_'+word.dep_,
                                        word.head.text))
                            sk.append((word.head.text, 
                                        word.dep_,
                                        word.text))
                    strt = end+1
                except:
                    break
            else:
                nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', 
                                           lang='en', use_gpu=use_gpu)
        return sk  
                
    def get_ns(self, btch_sz=1):
        return np.random.choice(self.data.neg_sample_array, self.ns*btch_sz)
    
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        u, d, v = zip(self.id_sk[idx])
        return [(u[0], v[0], d[0])]

    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _ in batch]
        all_v = [v for batch in batches for _, v, _ in batch]
        all_d = [d for batch in batches for _, _, d in batch]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_d)
