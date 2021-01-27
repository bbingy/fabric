import numpy as np
import os
from train import train
from inference import inference
from data_preprocess import data_prepare

if __name__ =='__main__':
    train_data_dir = '/home/lby/fabric/train3600.npy'
    train_X, train_Y = data_prepare(train_data_dir)

    val_data_dir = '/home/lby/fabric/val1200.npy'
    valid_X, valid_Y = data_prepare(val_data_dir)
    print(train_X.shape, valid_X.shape)

    test_data_dir = '/home/lby/fabric/test3600_nohole.npy'
    test_X, test_Y = data_prepare(test_data_dir)
    test_data_hole_dir = '/home/lby/fabric/test1200_hole.npy'
    test_X_hole, test_Y_hole = data_prepare(test_data_hole_dir)
    print(test_X.shape, test_X_hole.shape)

    print('Traing stage...')
    train(train_X, train_Y, valid_X, valid_Y)
    print('Testing stage...')
    pred_Y = inference(test_X, test_Y, False)
    pred_Y = inference(test_X_hole, test_Y_hole, True)