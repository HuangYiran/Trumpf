"""
run seq to seq model, use original metric 4 as score
"""
import torch
import pandas as pd
import shapely
import numpy as np
import sys
import torch
import argparse
import random
import math
import os

from sklearn.preprocessing import OneHotEncoder
from scipy.stats import stats
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

def main():
    num_epochs = 20
    batch_size = 30
    # set models and loss
    model = TS_rnn()
    #loss = PDLoss()
    loss = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    # set the scheduler
    lamb1 = lambda x: .1**(x//30)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lamb1)
    run('../data/', model, loss, optimizer, scheduler, num_epochs, batch_size, verbose = True)
    
def extract_data_from_pkl(path):
    df = pd.read_pickle(path)
    df['num_of_corr'] = df['List of Coordinates'].map(lambda x: len(x))
    # get onehot data for 'Teile-Nr'
    df = df.reset_index().iloc[:,1:]
    # get onehot data
    ohe = OneHotEncoder()
    tmp = ohe.fit_transform(df['num_of_corr'].values.reshape(-1, 1)).toarray()
    tmp = pd.DataFrame(tmp)
    tmp.columns = ['s1', 's2', 's3', 's4', 's5', 's6']
    df = pd.concat([df, tmp], axis = 1)
    # get onehot data for rotation
    df['Rot'][df['Rot'] == '5.00'] = '0.00'
    # get onehot data
    tmp = ohe.fit_transform(df['Rot'].values.reshape(-1, 1)).toarray()
    tmp = pd.DataFrame(tmp)
    tmp.columns = ['r1', 'r2']
    df = pd.concat([df, tmp], axis = 1)
    # change type of clumns for y
    df['Metric2'] = pd.to_numeric(df['Metric2'])
    df['Metric3'] = pd.to_numeric(df['Metric3'])
    df['Metric4'] = pd.to_numeric(df['Metric4'])
    # combine the sequence in one task and transform to numpy
    df['Jobid'] = df['Jobid'].map(lambda x: x.split('_')[0]+'_'+x.split('_')[1])

    x = []
    y = []
    for name, group in df.groupby('Jobid'):
        if len(group) == 46:
            x.append(group.iloc[:, 9:].values)
            y.append(group.iloc[:, 5:8].values)

    x = np.stack(x, axis = 0)
    y = np.stack(y, axis = 0)
    x = np.float32(x)
    y = np.float32(y)
    # distribute the data
    train_x = x[:int(len(x)*0.9)]
    test_x = x[int(len(x)*0.9):]
    train_y = y[:int(len(x)*0.9)]
    test_y = y[int(len(x)*0.9):]
    return train_x, train_y, test_x, test_y

def run(path, model, loss, optimizer, scheduler, num_epochs, batch_size, verbose = True):
    # list the pkl file in the path and only take the pkl files
    filenames = os.listdir(path)
    filenames = [filename for filename in filenames if filename.split('.')[-1] == 'pkl']
    # loda data
    for epoch in range(num_epochs):
        random.shuffle(filenames)
        for filename in filenames:
            train_x, train_y, test_x, test_y = extract_data_from_pkl(path + filename)
            traindata = Data(train_x, train_y)
            testdata = Data(train_x, train_y)
            dl_train = DataLoader(traindata, batch_size = batch_size, shuffle = True)
            dl_test = DataLoader(testdata, batch_size = batch_size, shuffle = True)
            train_model(dl_train, model, loss, scheduler, optimizer, epoch, batch_size, filename, verbose)
        test_lo = test_model(dl_test, model, loss)
        if verbose:
            # train loss
#            print('====> Epoch: {} Average train loss: {:.4f}'.format(
#                epoch,
#                train_loss/counter
#                ))
            # test loss
            print('====> Epoch: {} Average test loss: {:.4f}'.format(
                epoch,
                test_lo
                ))

def train_model(dl_train, model, loss, scheduler, optimizer, epoch, batch_size, filename, verbose = False):
    scheduler.step()
    model.train()
    train_loss = 0
    counter = 0
    for batch_idx, dat in enumerate(dl_train):
        counter += 1
        # train the model
        optimizer.zero_grad()
        inp, target = dat
        out = model(inp)
        lo = loss(out, target)
        lo.backward()
        optimizer.step()
        train_loss += lo.data
        if verbose:
            if batch_idx % 10 == 0:
                print('Train Epoch: {}\t Train file: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    filename,
                    batch_idx * batch_size,
                    batch_size * len(dl_train),
                    100.*batch_idx*batch_size/(batch_size * len(dl_train)),
                    lo.data
                    ))

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

##################
# model
##################
# bild the model, loss and data class
class TS_rnn(torch.nn.Module):
    """
    scores for each piece
    input:
        tensor size of (batch_size, seq_len, num_dim)
    output:
        tensor size of (batch_size, seq_len)
    """
    def __init__(self):
        super(TS_rnn, self).__init__()
        #change the structure of the network
        num_inp = 8
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
        return out.squeeze()

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
        self.data['train_x'] = self.add_file(x)
        self.data['train_y'] = self.add_file(y)[:, :, 0] # use the first metric tempately
        assert(len(self.data['train_x']) == len(self.data['train_y']))
        self.len = len(self.data['train_x'])

    def add_file(self, data):
        return torch.from_numpy(data)

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
        self.data['train_y'] = self.add_file(y)[:, :, 0] # use the first metric tempately
        self.data['train_y'] = torch.mean(self.data['train_y'], 1)
        assert(len(self.data['train_x']) == len(self.data['train_y']))
        self.len = len(self.data['train_x'])

    def add_file(self, data):
        return torch.from_numpy(data)

    def add_scores(self, path):
        return torch.FloatTensor([float(li.rstrip('\n')) for li in open(path)])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return (self.data['train_x'][index],
                self.data['train_y'][index])

if __name__ == '__main__':
    main()
