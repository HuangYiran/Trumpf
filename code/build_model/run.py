import torch
import pandas as pd
import numpy as np
from IncepLSTM import *
from CnnLSTM import *

def main():
    num_epochs = 200
    batch_size = 30
    # set models, loss and optimizer
    model = CnnLSTM()
    loss = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    run(model, '../data/train.pickle', '../data/matrix_feature.pickle', loss, optimizer, num_epochs, batch_size, True)


def run(model, traindata, polysdata, loss, optimizer, num_epochs = 200, batch_size = 32, n = 8, verbose = False):
    # run the model use one GPU only
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model
    # load data
    train = Data(traindata, polysdata)
    dl_train = DataLoader(train, batch_size = batch_size, shuffle = True)
    # train the model
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
            out = model(inp)
            lo = loss(out.squeeze(), target.squeeze())
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
        if epoch % 10 == 1:
            # save model pro 100 rounds
            torch.save(model.state_dict(), PATH)
        if verbose:
            # print train loss if verbose, without test result
            print('====> Epoch: {} Average train loss: {:.4f}'.format(
                epoch,
                train_loss/counter
                ))
            
if __name__ == '__main__':
    main()
