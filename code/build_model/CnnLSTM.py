import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class CnnLSTM(nn.Module):
    def __init__(self, num_classes = 1):
        super(CnnLSTM, self).__init__()
        num_cnno = 128
        num_hidden = 128
        self.cnn = nn.Sequential(
            BasicConv2d(1, 16, kernel_size=3, stride=2),
            BasicConv2d(16, 16, kernel_size=3),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            BasicConv2d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout2d())
        self.lstm = nn.LSTM(32, num_hidden, num_layers = 2, batch_first = True)
        self.fce = nn.Linear(num_hidden, num_classes)
    def forward(self, xs):
        """
        parameters:
        -------------
            xs: input data, list of pictures (with shape (N, 299, 299, 1))
        """
        co = []
        out = []
        seq_len = len(xs)
        batch_size, seq_len, width, height = xs.shape
        for i in range(seq_len):
            # (N, 299, 299, 1) => (N, num_hidden)
            #co.append(self.cnn(torch.flatten(x, 1)))
            print('%'*20)
            print('round: '+ str(i))
            tmp = self.cnn(xs[:,i,:,:].unsqueeze(1))
            co.append(tmp.squeeze())
            print(tmp.squeeze().shape)
        co = [i.unsqueeze(1) for i in co]
        co = torch.cat(co, 1)
        lo, _ = self.lstm(co)
        for i in range(seq_len):
            tmp = self.fce(lo[:, i,:])
            out.append(tmp)
        out = torch.cat(out, 1)
        return out.squeeze()


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        print(in_channels, out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Data():
    def __init__(self, x, y):
        self.data = {}
        self.data['data'] = self.add_file(x) # type of pandas 
        self.data['polys'] = self.add_file(y) # type of pandas
        self.jobs = self.data['data']['Jobid'].unique()
        self.len = len(self.jobs)
    
    def add_file(self, path):
        # read pickle file with pandas
        out = pd.read_pickle(path)
        out = out.astype({'Rot': 'float'})
        return out
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        # get item with jobid
        dat = self.data['data'][self.data['data']['Jobid'] == self.jobs[index]]
        # merge 
        dat = dat.merge(self.data['polys'], on=['bounding box', 'Rot'], how = 'left')[['matrix', 'score']]
        # transform and concatenate and split data 
        dat_x = np.concatenate(dat[['matrix']].values.tolist(), axis = 0) # shape of [50, 350, 350]
        dat_y = dat[['score']].values # shape of [50, 1]
        # transofrom to torch
        dat_x = torch.from_numpy(dat_x).type('torch.FloatTensor')
        dat_y = torch.from_numpy(dat_y).type('torch.FloatTensor')
        # move to gpu if exist
        if torch.cuda.is_available():
            dat_x = dat_x.cuda()
            dat_y = dat_y.cuda()
        return dat_x, dat_y