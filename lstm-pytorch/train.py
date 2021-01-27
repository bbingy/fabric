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
learning_rate = float(config.get('LstmSection', 'learning_rate'))
max_epoch = int(config.get('LstmSection', 'max_epoch'))
model_save_path = str(config.get('LstmSection', 'model_save_path'))
test_save_path = str(config.get('LstmSection', 'test_save_path'))
val_save_path = str(config.get('LstmSection', 'val_save_path'))
model_name = str(config.get('LstmSection', 'model_name'))
patience = int(config.get('LstmSection', 'patience'))

def train(train_X, train_Y, valid_X, valid_Y):
    train_X, train_Y = torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).float()
    valid_X, valid_Y = torch.from_numpy(valid_X).float(), torch.from_numpy(valid_Y).float()

    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")

    if model_name == 'lstm':
        model = VanillaLSTM().to(device)
    if model_name == 'talstm':
        model = TALSTM().to(device)
    if model_name == 'd2talstm':
        model = D2TALSTM().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    valid_loss_min = float("inf")
    train_loss_valid = float("inf")
    bad_epoch = 0
    global_step = 0
    train_bs_epoch = int(train_X.shape[0]/batch_size)
    valid_bs_epoch = int(valid_X.shape[0]/batch_size)
    train_X, train_Y = train_X.permute(0,2,1).to(device), train_Y.to(device)
    valid_X, valid_Y = valid_X.permute(0,2,1).to(device), valid_Y.to(device)

    for epoch in range(max_epoch):
        print("Epoch {}/{}".format(epoch, max_epoch))
        model.train()
        train_loss_array = []
        for i in range(train_bs_epoch):
            _train_X = train_X[i*batch_size:i*batch_size+batch_size,:,:]
            _train_Y = train_Y[i*batch_size:i*batch_size+batch_size,:]
            optimizer.zero_grad()
            pred_Y = model(_train_X)
            loss = criterion(pred_Y, _train_Y)
            loss.backward()
            optimizer.step()
            train_loss_array.append(loss.item())
            global_step += 1

        # 以下为早停机制，当模型训练连续config.patience个epoch都没有使验证集预测效果提升时，就停止，防止过拟合
        model.eval()
        valid_loss_array = []
        # hidden_valid = None
        for i in range(valid_bs_epoch):
            _valid_X = valid_X[i*batch_size:i*batch_size+batch_size,:,:]
            _valid_Y = valid_Y[i*batch_size:i*batch_size+batch_size,:]
            pred_Y = model(_valid_X)
            # if not config.do_continue_train: hidden_valid = None
            # hidden_valid = None
            loss = criterion(pred_Y, _valid_Y)
            valid_loss_array.append(loss.item())

        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)
        print("The train loss is {:.6f}. ".format(train_loss_cur) +
            "The valid loss is {:.6f}.".format(valid_loss_cur))
        if epoch>6:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate/2)

        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            train_loss_valid = train_loss_cur
            bad_epoch = 0
            torch.save(model.state_dict(), model_save_path + model_name + str(seq_length) + '.pth')  # 模型保存
        else:
            bad_epoch += 1
            if bad_epoch >= patience:    # 如果验证集指标连续patience个epoch没有提升，就停掉训练
                print("The min valid loss is {:.6f}. ".format(valid_loss_min) +
                "While the train loss is {:.6f}.".format(train_loss_valid))
                break

if __name__ == '__main__':
    ker_size = 5
    seq_length = 50

    train_data_dir = '/home/lby/fabric/train3600.npy'
    data = np.load(train_data_dir)
    data = normalize(data)
    data = meanfilter(data)
    train_X, train_Y = datatoseq(data)

    val_data_dir = '/home/lby/fabric/val1200.npy'
    data = np.load(val_data_dir)
    data = normalize(data)
    data = meanfilter(data)
    valid_X, valid_Y = datatoseq(data)
    print(train_X.shape, valid_X.shape)

    train(train_X, train_Y, valid_X, valid_Y)