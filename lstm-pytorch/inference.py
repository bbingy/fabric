import numpy as np
import os
from data_preprocess import normalize, meanfilter, datatoseq
from model_lstm import VanillaLSTM, TALSTM, D2TALSTM
import configparser
import torch

config = configparser.ConfigParser()
path = os.path.split(os.path.realpath(__file__))[0] + '/model.conf'
config.read(path)
batch_size = int(config.get('LstmSection', 'batch_size'))
use_cuda = config.get('LstmSection', 'use_cuda')
model_save_path = str(config.get('LstmSection', 'model_save_path'))
test_save_path = str(config.get('LstmSection', 'test_save_path'))
val_save_path = str(config.get('LstmSection', 'val_save_path'))
model_name = str(config.get('LstmSection', 'model_name'))

def inference(test_X, test_y, ishole):
    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")

    test_X = torch.from_numpy(test_X).float()
    test_X = test_X.permute(0,2,1).to(device)
    
    # 加载模型
    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
    if model_name == 'lstm':
        model = VanillaLSTM().to(device)
    if model_name == 'talstm':
        model = TALSTM().to(device)
    if model_name == 'd2talstm':
        model = D2TALSTM().to(device)
    model.load_state_dict(torch.load(model_save_path + model_name + str(seq_length) + '.pth'))   # 加载模型参数

    test_bs_epoch = int(test_X.shape[0]/batch_size)
    result = np.empty([test_X.shape[0], test_X.shape[2]])

    # 预测过程
    model.eval()
    test_loss_array = []
    for i in range(test_bs_epoch):
        data_X = test_X[i*batch_size:i*batch_size+batch_size,:,:]
        pred_X = model(data_X) # (10,1920)
        result[i*batch_size:i*batch_size+batch_size,:] = pred_X.detach().cpu().numpy()
    test_loss = np.mean((result-test_Y)**2)
    print("The test loss is {:.6f}. ".format(test_loss))
    if ishole == False:
        np.save(test_save_path + model_name + str(seq_length) + '.npy', result)
    else:
        np.save(test_save_path + model_name + str(seq_length) + '_hole.npy', result)

    return result

if __name__ == '__main__':
    test_data_dir = '/home/lby/fabric/test1200_hole.npy'
    data = np.load(test_data_dir)
    data = normalize(data)
    data = meanfilter(data)
    test_X, test_Y = datatoseq(data)
    pred_Y = inference(test_X, test_Y)
    print(pred_Y.shape)