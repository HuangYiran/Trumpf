# prepare for neural network training
import torch
import argparse
import pandas as pd
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help = 'the path of the training data', default = '../results/Dataframe_feature_6_area_quote.pkl')
parser.add_argument('-o', "--output", help = 'the path to save the output file', default = '../data/')

def main():
    args = parser.parse_args()
    print(args.data)
    #result = pd.read_pickle('../results/Dataframe_feature_6_area_quote.pkl')
    result = pd.read_pickle(args.data)
    train_x = result.iloc[:int(len(result)*0.9), :650].values.reshape(int(len(result)*0.9), 50, -1)
    test_x = result.iloc[int(len(result)*0.9):, :650].values.reshape(len(result) - int(len(result)*0.9), 50, -1)
    train_y = result.iloc[:int(len(result)*0.9), 650:].values.reshape(int(len(result)*0.9), 50, -1)
    test_y = result.iloc[int(len(result)*0.9):, 650:].values.reshape(len(result) - int(len(result)*0.9), 50, -1)
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    # save to the local system
    np.save(args.output + 'rnn_train_x', train_x)
    np.save(args.output + 'rnn_train_y', train_y)
    np.save(args.output + 'rnn_test_x', test_x)
    np.save(args.output + 'rnn_test_y', test_y)

if __name__ == '__main__':
    main()
