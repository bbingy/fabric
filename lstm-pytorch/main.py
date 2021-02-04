import numpy as np
import os
import argparse
from train import train
from inference import inference
from data_preprocess import data_prepare

if __name__ =='__main__':
    desc = 'train a model for fabric defect detection'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-seq_length', help='sequence length for model',
                        default=200, type=int)
    # parser.add_argument('-m', help='market name', default='NASDAQ')
    # parser.add_argument('-t', help='fname for selected tickers')
    args = parser.parse_args()

    train_data_dir = '/home/lby/fabric/train3600.npy'
    train_X, train_Y = data_prepare(train_data_dir, args.seq_length)

    val_data_dir = '/home/lby/fabric/val1200.npy'
    valid_X, valid_Y = data_prepare(val_data_dir, args.seq_length)
    print(train_X.shape, valid_X.shape)

    test_data_dir = '/home/lby/fabric/test3600_nohole.npy'
    test_X, test_Y = data_prepare(test_data_dir, args.seq_length)
    np.save('/home/lby/fabric/test3600_gt_30.npy', test_Y)

    test_data_hole_dir = '/home/lby/fabric/test1200_hole.npy'
    test_X_hole, test_Y_hole = data_prepare(test_data_hole_dir, args.seq_length)
    np.save('/home/lby/fabric/test1200_gt_30.npy', test_Y_hole)
    print(test_X.shape, test_X_hole.shape)

    print('Traing stage...')
    train(train_X, train_Y, valid_X, valid_Y, args.seq_length)
    print('Testing stage...')
    print('Test no hole sample')
    pred_Y = inference(test_X, test_Y, False, args.seq_length)
    print('Test hole sample')
    pred_Y = inference(test_X_hole, test_Y_hole, True, args.seq_length)