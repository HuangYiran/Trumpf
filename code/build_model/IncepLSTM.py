import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import time
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def main():
    root = '/smartdata/hj7422/Documents/Workplace/Trumpf/'
    num_epochs = 20
    batch_size = 10
    # set models, loss and optimizer
    #model = CnnLSTM()
    model = IncepLSTM(n=16)
    loss = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    for i in range(100):
        run(model, root + 'data/train.pickle', root + 'data/matrix_feature.pickle', loss, optimizer, num_epochs, batch_size, i, True)

def run(model, traindata, polysdata, loss, optimizer, num_epochs = 200, batch_size = 32, n = 8, verbose = False):
    root = '/smartdata/hj7422/Documents/Workplace/Trumpf/'
    # run the model use one GPU only
    if torch.cuda.is_available():
        # model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    else:
        model = model
    # load data
    train = Data(traindata, polysdata)
    dl_train = DataLoader(train, batch_size = batch_size, shuffle = True)
    # train the model
    print('start to train model')
    for epoch in range(num_epochs):
        #scheduler.step()
        model.train()
        train_loss = 0
        counter = 0
        for batch_idx, dat in enumerate(dl_train):
            counter += 1
            # train the model
            optimizer.zero_grad()
            inp, target = dat
            if inp.shape[0] < 2:
                continue
            #print('run model')
            #print(time.time())
            out = model(inp)
            lo = loss(out.squeeze(), target.squeeze())
            lo.backward()
            optimizer.step()
            train_loss += float(lo.data)
            if verbose:
                if batch_idx % 10 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.16f}'.format(
                        epoch,
                        batch_idx * batch_size,
                        len(train),
                        100.*batch_idx*batch_size/len(train),
                        lo.data,
                        time.time()
                        ))
        if epoch % 10 == 1:
            # save model pro 100 rounds
            torch.save(model.state_dict(), root + 'models/IncepLSTM/inceplstm_'+ str(n) + '_' + str(epoch))
        #torch.save(model.state_dict(), root + 'models/CnnLSTM/cnnlstm_' + str(epoch))

        if verbose:
            # print train loss if verbose, without test result
            print('====> Epoch: {} Average train loss: {:.4f}'.format(
                epoch,
                train_loss/counter
                ))
 
class IncepLSTM(nn.Module):
    def __init__(self, num_classes = 1, n = 2):
        super(IncepLSTM, self).__init__()
        num_cnno = 128
        num_hidden = 128
        self.cnn = nn.Sequential(
            BasicConv2d(1, int(16/n), kernel_size=3, stride=2),
            BasicConv2d(int(16/n), int(16/n), kernel_size=3),
            BasicConv2d(int(16/n), int(32/n), kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            BasicConv2d(int(32/n), int(40/n), kernel_size=1),
            BasicConv2d(int(40/n), int(96/n), kernel_size=3),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            InceptionA(int(96/n), pool_features=16, n = n), # 224/n + pool_features
            InceptionA(int(224/n + 16), pool_features=32, n = n),
            InceptionA(int(224/n + 32), pool_features=32, n = n),
            InceptionB(int(224/n + 32), n = n), # 480/n + c_in
            InceptionC(int(480/n + 224/n + 32), channels_7x7=64, n = n), # 768/n
            InceptionC(int(768/n), channels_7x7=80, n = n),
            InceptionC(int(768/n), channels_7x7=80, n = n),
            InceptionC(int(768/n), channels_7x7=80, n = n),
            InceptionD(int(768/n), n = n), # 512/n + c_in
            InceptionE(int(768/n + 512/n), n = n), # (320 + 384*4 + 192)/n
            InceptionE(int(2048/n), n = n),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout2d())
        self.lstm = nn.LSTM(int(2048/n), num_hidden, num_layers = 2, batch_first = True)
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
            #print('%'*20)
            #print('round: '+ str(i))
            tmp = self.cnn(xs[:,i,:,:].unsqueeze(1))
            co.append(tmp.squeeze())
            #print(tmp.squeeze().shape)
        co = [i.unsqueeze(1) for i in co]
        co = torch.cat(co, 1)
        lo, _ = self.lstm(co)
        for i in range(seq_len):
            tmp = self.fce(lo[:, i,:])
            out.append(tmp)
        out = torch.cat(out, 1)
        return out.squeeze()
    
# layer: number of output channel 
# InceptionA(in_channels, pool_features): 224/n + pool_features; H, W
# InceptionB(in_channels): 480/n + in_channels; H/2, W/2
# InceptionC(in_channels, channels_7x7): 768/n; H, W
# InceptionD(in_channels): 512/n + in_channels; H/2, W/2
# InceptionE(in_channels): 1280/n; H, W
# InceptionAux(in_channels, pool_features):
# BasicConv2d(in_channels, out_channels): Conv2d + BN
class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features, n = 2):
        """
        n = 1, 2, 4, 6
        """
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, int(64/n), kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, int(48/n), kernel_size=1)
        self.branch5x5_2 = BasicConv2d(int(48/n), int(64/n), kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, int(64/n), kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(int(64/n), int(96/n), kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(int(96/n), int(96/n), kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        #print('InceptionA')
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels, n = 2):
        """
        n = 1, 2, 4, 6
        """
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, int(384/n), kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, int(64/n), kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(int(64/n), int(96/n), kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(int(96/n), int(96/n), kernel_size=3, stride=2)

    def forward(self, x):
        #print('InceptionB')
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7, n = 2):
        """
        n = 1, 2, 4, 6
        """
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, int(192/n), kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, int(192/n), kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, int(192/n), kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, int(192/n), kernel_size=1)

    def forward(self, x):
        #print('InceptionC')
        branch1x1 = self.branch1x1(x)
        #print('InceptionC - 1')
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        #print('InceptionC - 2')
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        #print('InceptionC - 3')
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        #print('InceptionC - 4')
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels, n = 2):
        """
        n = 1, 2, 4, 6
        """
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, int(192/n), kernel_size=1)
        self.branch3x3_2 = BasicConv2d(int(192/n), int(320/n), kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, int(192/n), kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(int(192/n), int(192/n), kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(int(192/n), int(192/n), kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(int(192/n), int(192/n), kernel_size=3, stride=2)

    def forward(self, x):
        #print('InceptionD')
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels, n = 2):
        """
        n = 1, 2, 4, 6
        """
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, int(320/n), kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, int(384/n), kernel_size=1)
        self.branch3x3_2a = BasicConv2d(int(384/n), int(384/n), kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(int(384/n), int(384/n), kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, int(448/n), kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(int(448/n), int(384/n), kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(int(384/n), int(384/n), kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(int(384/n), int(384/n), kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, int(192/n), kernel_size=1)

    def forward(self, x):
        #print('InceptionE')
        branch1x1 = self.branch1x1(x)
        #print('InceptionE - 1')
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)
        #print('InceptionE - 2')
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        #print('InceptionE - 3')
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        #print(branch1x1.shape, branch3x3.shape, branch3x3dbl.shape, branch_pool.shape)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, n = 2):
        """
        n = 1, 2, 4, 6
        """
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, int(128/n), kernel_size=1)
        self.conv1 = BasicConv2d(int(128/n), int(768/n), kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(int(768/n), num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


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
        self._combine()
        self.len = len(self.data['Xs'])
    
    def add_file(self, path):
        # read pickle file with pandas
        out = pd.read_pickle(path)
        out = out.astype({'Rot': 'float'})
        #return out.iloc[:200000, :]
        return out
    
    def _combine(self):
        jobs = self.data['data']['Jobid'].unique()
        # shuffle
        random.shuffle(jobs)
        Xs = []
        ys = []
        counter = 0
        for i in jobs:
            if counter > 700:
                break
            dat = self.data['data'][self.data['data']['Jobid'] == i]
            dat = dat.merge(self.data['polys'], on=['bounding box', 'Rot'], how = 'left')[['matrix', 'score']]
            dat_x = np.concatenate(dat[['matrix']].values.tolist(), axis = 0) # shape of [50, 350, 350]
            dat_y = dat[['score']].values # shape of [50, 1]
            dat_x = torch.from_numpy(dat_x).type('torch.FloatTensor')
            dat_y = torch.from_numpy(dat_y).type('torch.FloatTensor')
            if torch.cuda.is_available():
                dat_x = dat_x.cuda()
                dat_y = dat_y.cuda()
            Xs.append(dat_x)
            ys.append(dat_y)
            counter += 1
        del self.data['data']
        self.data['Xs'] = Xs
        self.data['ys'] = ys

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.data['Xs'][index], self.data['ys'][index]


class Data2():
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

if __name__ == '__main__':
    main()
