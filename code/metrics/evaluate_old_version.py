import pandas as pd
import shapely
import numpy as np
import sys
import torch
import argparse
import random
import math
import os
import warnings
from gensim.models import word2vec
from shapely.geometry import LineString, Polygon
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler# 好处在于可以保存训练集中的参数（均值、方差）
from scipy.stats import stats
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import tqdm
import gc

###########################################################################################
# run evaluate wich command                                                               #
# tmp = evaluate('../models/TS_rnn_mean/rnn_15.pkl', '../results/sample_test_6_mean.pkl') #
###########################################################################################
def evaluate(pm, testdata):
    # read extra test set
    test = pd.read_pickle(testdata)
    test_x = test.iloc[:, :650].values.reshape(len(test), 50, -1)
    test_y = test.iloc[:, 650:].values.reshape(len(test), 50, -1)
    np.save('../data/rnn_test_x', test_x)
    np.save('../data/rnn_test_y', test_y)
    loss = torch.nn.L1Loss()
    test_x = '../data/rnn_test_x.npy'
    test_y = '../data/rnn_test_y.npy'
    test = Data(test_x, test_y)
    dl_test = DataLoader(test, batch_size = 100, shuffle = True)
    model = torch.load(pm)
    me = 0
    for i in range(5):
        lo = significant_test(dl_test, model, loss)
        me += lo
        print('test ' + str(i)+': ' + str(lo))
    hit_count = metric3(dl_test, model, loss)
    hit_count = sorted(hit_count.items(), key = lambda x: x[0])
    #out = [i[1] for i in hit_count]
    out = hit_count
    return (me, out)

# write the test function
def test_model(dl_test, model, loss):
    model.eval()
    test_loss = 0
    counter = 0
    for batch_idx, dat in enumerate(dl_test):
        counter += 1
        # codes to be changed
        inp, target = dat
        out = model(inp)
        lo = loss(out, target)
        test_loss += lo.data
    return test_loss/counter

def significant_test(dl_test, model, loss):
    model.eval()
    test_loss = 0
    counter = 0
    hit = 0
    miss = 0
    for batch_idx, dat in enumerate(dl_test):
        counter += 1
        # codes to be changed
        inp, target = dat
        out = model(inp)
        #target = target.mean(dim = 1)
        target = target[:, :].mean(dim = 1)
        #print(out.shape)
        #out = out.mean(dim = 1)
        out = out[:, :].mean(dim = 1)
        #print(out.shape)
        if len(inp) > 5:
            _, top_target = torch.topk(target, 1, largest=False)
            _, top_predict = torch.topk(out, 5, largest = False)
            if top_target in top_predict:
                hit += 1
            else:
                miss += 1
    return hit * 1.0/(hit + miss)

def metric2(dl_test, model, loss):
    model.eval()
    counter = 0
    hit_count = {}
    for batch_idx, dat in enumerate(dl_test):
        counter += 1
        inp, target = dat
        out = model(inp)
        #target = target.mean(dim = 1)
        #out = out.mean(dim = 1)
        target = target[:, :].mean(dim = 1)
        out = out[:, :].mean(dim = 1)
        if len(inp) > 5:
            _, index_top_target = torch.topk(target, 1, largest = False)
            _, index_rank = torch.topk(out, len(target), largest = False)
            index_rank = index_rank.tolist()
            index_in_rank = index_rank.index(index_top_target)
            if index_in_rank not in hit_count.keys():
                #print('create new key')
                hit_count[index_in_rank] = 1
            else:
                #print('add one')
                hit_count[index_in_rank] = hit_count[index_in_rank] + 1
    return hit_count

def metric3(dl_test, model, loss):
    model.eval()
    counter = 0
    hit_count = {}
    for batch_idx, dat in enumerate(dl_test):
        counter += 1
        inp, target = dat
        out = model(inp)
        #target = target.mean(dim = 1)
        #out = out.mean(dim = 1)
        target = target[:, :].mean(dim = 1)
        out = out[:, :].mean(dim = 1)
        if len(inp) > 5:
            #_, index_top_target = torch.topk(target, 1, largest = False)
            _, index_top_out = torch.topk(out, 1, largest = False)
            #_, index_rank = torch.topk(out, len(target), largest = False)
            _, index_rank = torch.topk(target, len(target), largest = False)
            index_rank = index_rank.tolist()
            index_in_rank = index_rank.index(index_top_out)
            if index_in_rank not in hit_count.keys():
                #print('create new key')
                hit_count[index_in_rank] = 1
            else:
                #print('add one')
                hit_count[index_in_rank] = hit_count[index_in_rank] + 1
    return hit_count



"""
bild the model, loss and data class, including two different versions
version 1:
seq to seq model
version old:
the old seq to seq model without any paramseters
versioin 2:
seq to 1 model
"""
class TS_rnn(torch.nn.Module):
    """
    scores for each piece
    input:
        tensor size of (batch_size, seq_len, num_dim)
    output:
        tensor size of (batch_size, seq_len)
    """
    def __init__(self, num_hidden = 64, num_layers = 2, dropout = 0.5):
        super(TS_rnn, self).__init__()
        #change the structure of the network
        num_inp = 13
        self.rnn = torch.nn.LSTM(input_size = num_inp, hidden_size = num_hidden, num_layers = num_layers, dropout = dropout)
        self.mlp = torch.nn.Sequential(
                torch.nn.Linear(num_hidden, 16),
                torch.nn.Dropout(),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1)
                )

    def forward(self, inp):
        # input of the rnn (seq_len, batch, input_size)
        data_in = torch.transpose(inp, 0, 1)
        # run rnn, it has two output
        out_rnn, _ = self.rnn(data_in)
        out_rnn = torch.transpose(out_rnn, 0, 1) # (batch_size, seq_len, num_dim)
        # rnn the mlp
        batch_size, seq_len, num_dim = out_rnn.shape
        out = []
        for i in range(seq_len):
            tmp = self.mlp(out_rnn[:, i,:])
            out.append(tmp)
        # now out is list of (batch_size, 1), combine the items in the list to get the output with size (batch_size, seq_len)
        out = torch.cat(out, 1)
        #return out.squeeze() when the batch_size == 1, this can course trouble
        return out

class TS_rnn_old(torch.nn.Module):
    """
    scores for each piece
    input:
        tensor size of (batch_size, seq_len, num_dim)
    output:
        tensor size of (batch_size, seq_len)
    """
    def __init__(self):
        super(TS_rnn_old, self).__init__()
        #change the structure of the network
        num_inp = 13
        num_hidden = 64
        self.rnn = torch.nn.LSTM(input_size = num_inp, hidden_size = num_hidden, num_layers = 2)
        self.mlp = torch.nn.Sequential(
                torch.nn.Linear(num_hidden, 16),
                torch.nn.Dropout(),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1)
                )

    def forward(self, inp):
        # input of the rnn (seq_len, batch, input_size)
        data_in = torch.transpose(inp, 0, 1)
        # run rnn, it has two output
        out_rnn, _ = self.rnn(data_in)
        out_rnn = torch.transpose(out_rnn, 0, 1) # (batch_size, seq_len, num_dim)
        # rnn the mlp
        batch_size, seq_len, num_dim = out_rnn.shape
        out = []
        for i in range(seq_len):
            tmp = self.mlp(out_rnn[:, i,:])
            out.append(tmp)
        # now out is list of (batch_size, 1), combine the items in the list to get the output with size (batch_size, seq_len)
        out = torch.cat(out, 1)
        #return out.squeeze() when the batch_size == 1, this can course trouble
        return out

class TS_rnn2(torch.nn.Module):
    """
    scores only for the whole task
    input:
        tensor size of (batch_size, seq_len, num_dim)
    output:
        tensor size of (batch_size)
    """
    def __init__(self):
        super(TS_rnn2, self).__init__()
        #change the structure of the network
        num_inp = 8
        num_hidden = 64
        self.rnn = torch.nn.LSTM(input_size = num_inp, hidden_size = num_hidden, num_layers = 2)
        self.mlp = torch.nn.Sequential(
                torch.nn.Linear(num_hidden, 64),
                torch.nn.Dropout(),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1)
                )

    def forward(self, inp):
        # input of the rnn (seq_len, batch, input_size)
        data_in = torch.transpose(inp, 0, 1)
        # run rnn, it has two output
        out_rnn, _ = self.rnn(data_in)
        out_rnn = torch.transpose(out_rnn, 0, 1) # (batch_size, seq_len, num_dim)
        # only use the last output
        out_rnn = out_rnn[:, -1, :].squeeze()
        # rnn the mlp
        out = self.mlp(out_rnn)
        return out.squeeze()
    
class PDLoss(torch.nn.Module):
    def __init__(self, p = 2):
        super(PDLoss, self).__init__()
        self.pd = torch.nn.PairwiseDistance(p)

    def forward(self, o, t):
        # out: (batch_size, 1)
        out = self.pd(o, t)
        return out.mean()

class Data:
    """
    data class for TS_rnn
    """
    def __init__(self, x, y):
        self.data = {}
        self.data['train_x'] = self.add_file(x).float()
        self.data['train_y'] = self.add_file(y)[:, :, -1].float() # use the first metric tempately
        assert(len(self.data['train_x']) == len(self.data['train_y']))
        self.len = len(self.data['train_x'])

    def add_file(self, path):
        return torch.from_numpy(np.load(path))

    def add_scores(self, path):
        return torch.FloatTensor([float(li.rstrip('\n')) for li in open(path)])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.data['train_x'][index],
                self.data['train_y'][index])

class Data2:
    """
    data class for TS_rnn2
    """
    def __init__(self, x, y):
        self.data = {}
        self.data['train_x'] = self.add_file(x)
        self.data['train_y'] = self.add_file(y)[:, :, -1] # use the first metric tempately
        self.data['train_y'] = torch.mean(self.data['train_y'], 1)
        assert(len(self.data['train_x']) == len(self.data['train_y']))
        self.len = len(self.data['train_x'])

    def add_file(self, path):
        return torch.from_numpy(np.load(path))

    def add_scores(self, path):
        return torch.FloatTensor([float(li.rstrip('\n')) for li in open(path)])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.data['train_x'][index],
                self.data['train_y'][index])
