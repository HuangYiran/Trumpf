"""
run seq to seq model, use original metric 4 + high + area quote as score
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
    # set path
    root = '/smartdata/hj7422/Documents/Workplace/Trumpf/'
    train_x = root + 'data/rnn_train_x.npy'
    train_y = root + 'data/rnn_train_y.npy'
    test_x = root + 'data/rnn_test_x.npy'
    test_y = root + 'data/rnn_test_y.npy'
    # set hype params
    num_epochs = 2000
    batch_size = 32
    # set models and loss
    model = TS_rnn(num_hidden = 64, num_layers = 2, dropout = 0.5)
    loss = torch.nn.L1Loss()
    # because of the scheduler the original learning rate should be set to a relative large value. e.g. 1.0
    optimizer = torch.optim.Adam(model.parameters(), lr = 1)
    # set the scheduler
    lamb1 = lambda x: 0.01 * 0.1**((x%100)//30)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lamb1)
    run(train_x, train_y, test_x, test_y, model, loss, optimizer, scheduler, num_epochs, batch_size, root, verbose = True)
    

def run(train_x, train_y, test_x, test_y, model, loss, optimizer, scheduler, num_epochs, batch_size, root = './', verbose = True):
    # loda data
    train = Data(train_x, train_y)
    test = Data(test_x, test_y)
    dl_train = DataLoader(train, batch_size = batch_size, shuffle = True)
    dl_test = DataLoader(test, batch_size = batch_size, shuffle = True)
    # data to be collectd
    test_loss = []
    test_sig = []
    # weight to be collected: num_layer need to be larger than 2
    mod_wh_l0 = []
    mod_wh_l1 = []
    mod_wi_l0 = []
    mod_wi_l1 = []
    mean_mod_wh_l0 = []
    mean_mod_wh_l1 = []
    mean_mod_wi_l0 = []
    mean_mod_wi_l1 = []
    std_mod_wh_l0 = []
    std_mod_wh_l1 = []
    std_mod_wi_l0 = []
    std_mod_wi_l1 = []
    # train the model
    for epoch in range(num_epochs):
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
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch,
                        batch_idx * batch_size,
                        len(train),
                        100.*batch_idx*batch_size/len(train),
                        lo.data
                        ))
        test_lo = test_model(dl_test, model, loss)
        hit_rate = significant_test(dl_test, model, loss)
        # collect data
        test_loss.append(test_lo)
        test_sig.append(hit_rate)
        mod_wh_l0.append(model.rnn.weight_hh_l0)
        mod_wh_l1.append(model.rnn.weight_hh_l1)
        mod_wi_l0.append(model.rnn.weight_ih_l0)
        mod_wi_l1.append(model.rnn.weight_ih_l1)
        # just for test
        # save model
        if epoch % 10 == 0:
            # save model
            torch.save(model, root + 'models/rnn_' + str(epoch) + '.pkl')
            # save result
            torch.save(test_loss, root + 'results/test_loss.pkl')
            torch.save(test_sig, root + 'results/test_sig.pkl')
            # save weight
            mod_wh_l0 = torch.stack(mod_wh_l0, dim = 0)
            mod_wh_l1 = torch.stack(mod_wh_l1, dim = 0)
            mod_wi_l0 = torch.stack(mod_wi_l0, dim = 0)
            mod_wi_l1 = torch.stack(mod_wi_l1, dim = 0)
            mean_mod_wh_l0.append(mod_wh_l0.mean(dim = 0))
            mean_mod_wh_l1.append(mod_wh_l1.mean(dim = 0))
            mean_mod_wi_l0.append(mod_wi_l0.mean(dim = 0))
            mean_mod_wi_l1.append(mod_wi_l1.mean(dim = 0))
            std_mod_wh_l0.append(mod_wh_l0.std(dim = 0))
            std_mod_wh_l1.append(mod_wh_l1.std(dim = 0))
            std_mod_wi_l0.append(mod_wh_l0.std(dim = 0))
            std_mod_wi_l1.append(mod_wh_l1.std(dim = 0))
            torch.save(mean_mod_wh_l0, root + 'results/mean_mod_wh_l0')
            torch.save(mean_mod_wh_l1, root + 'results/mean_mod_wh_l1')
            torch.save(mean_mod_wi_l0, root + 'results/mean_mod_wi_l0')
            torch.save(mean_mod_wi_l1, root + 'results/mean_mod_wi_l1')
	    torch.save(std_mod_wh_l0, root + 'results/std_mod_wh_l0')
            torch.save(std_mod_wh_l1, root + 'results/std_mod_wh_l1')
            torch.save(std_mod_wi_l0, root + 'results/std_mod_wi_l0')
            torch.save(std_mod_wi_l1, root + 'results/std_mod_wi_l1')

            mod_wh_l0 = []
            mod_wh_l1 = []
            mod_wi_l0 = []
            mod_wi_l1 = []
        if verbose:
            # train loss
            print('====> Epoch: {} Average train loss: {:.4f}'.format(
                epoch,
                train_loss/counter
                ))
            # test loss
            print('====> Epoch: {} Average test loss: {:.4f}'.format(
                epoch,
                test_lo
                ))
            # significant test
            print('====> Epoch: {} Average hit rate in 10 candidate: {: .4f}'.format(
                epoch,
                hit_rate
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
        target = target.mean(dim = 1)
        #print(out.shape)
        out = out.mean(dim = 1)
        #print(out.shape)
        if len(inp) > 5:
            _, top_target = torch.topk(target, 1)
            _, top_predict = torch.topk(out, 5)
            if top_target in top_predict:
                hit += 1
            else:
                miss += 1
    return hit * 1.0/(hit + miss)

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
    def __init__(self, num_hidden = 64, num_layers = 2, dropout = 0.5):
        super(TS_rnn, self).__init__()
        #change the structure of the network
        num_inp = 17
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

if __name__ == '__main__':
    main()

