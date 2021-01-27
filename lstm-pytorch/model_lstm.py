import torch
from torch.nn import Module, LSTM, Linear, Conv1d, Dropout, ReLU, MaxPool1d, Flatten, LeakyReLU, Softmax
from torch.utils.data import DataLoader, TensorDataset
import configparser
import os

config = configparser.ConfigParser()
path = os.path.split(os.path.realpath(__file__))[0] + '/model.conf'
config.read(path)
input_size = int(config.get('LstmSection', 'input_size'))
lstm_hidden_size = int(config.get('LstmSection', 'hidden_size'))
lstm_layers = int(config.get('LstmSection', 'lstm_layers'))
lstm_output_size = int(config.get('LstmSection', 'output_size'))
batch_size = int(config.get('LstmSection', 'batch_size'))

class VanillaLSTM(Module):
    def __init__(self):
        super(VanillaLSTM, self).__init__()
        self.lstm = LSTM(input_size=input_size, hidden_size=lstm_hidden_size,
                         num_layers=lstm_layers, batch_first=True)
        self.linear = Linear(in_features=lstm_hidden_size, out_features=lstm_output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        linear_out = self.linear(lstm_out[:,-1,:])
        return linear_out


class TALSTM(Module):
    def __init__(self):
        super(TALSTM, self).__init__()
        self.lstm = LSTM(input_size=input_size, hidden_size=lstm_hidden_size,
                         num_layers=lstm_layers, batch_first=True)
        self.dense2 = Linear(in_features=batch_size, out_features=1)
        self.leaky_relu2 = LeakyReLU(0.2)
        self.dense3 = Linear(in_features=batch_size, out_features=1)
        self.leaky_relu3 = LeakyReLU(0.2)
        self.softmax = Softmax(dim=0)
        self.linear = Linear(in_features=lstm_hidden_size*2, out_features=lstm_output_size)

        self.all_one = torch.ones((lstm_hidden_size*2, 1), dtype=torch.float32).cuda()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        feature = lstm_out[:,-1,:]
        feature = feature.permute(1,0)

        head_weight = self.dense2(feature)
        head_weight = self.leaky_relu2(head_weight)
        tail_weight = self.dense3(feature)
        tail_weight = self.leaky_relu3(tail_weight)
        weight = torch.matmul(head_weight, self.all_one.permute(1,0)) + \
               torch.matmul(self.all_one, tail_weight.permute(1,0)) + rel_weight[:,:,-1]
        
        weight = self.softmax(weight)
        outputs_proped = torch.matmul(weight, feature)
        outputs_concated = torch.cat([feature, outputs_proped], 0)

        linear_out = self.linear(outputs_concated.permute(1,0))
        return linear_out

class D2TALSTM(Module):
    def __init__(self):
        super(D2TALSTM, self).__init__()
        self.lstm = LSTM(input_size=input_size, hidden_size=lstm_hidden_size,
                         num_layers=lstm_layers, batch_first=True)
        self.lstm1 = LSTM(input_size=input_size, hidden_size=lstm_hidden_size,
                         num_layers=lstm_layers, batch_first=True)
        self.lstm2 = LSTM(input_size=input_size, hidden_size=lstm_hidden_size,
                         num_layers=lstm_layers, batch_first=True)
        self.dense2 = Linear(in_features=batch_size, out_features=1)
        self.leaky_relu2 = LeakyReLU(0.2)
        self.dense3 = Linear(in_features=batch_size, out_features=1)
        self.leaky_relu3 = LeakyReLU(0.2)
        self.softmax = Softmax(dim=0)
        self.linear = Linear(in_features=lstm_hidden_size*2*3, out_features=lstm_output_size)

        self.all_one = torch.ones((lstm_hidden_size*3, 1), dtype=torch.int).cuda()

    def forward(self, x):
        lstm_out, _ = self.lstm(x) # (10,200,1920)
        feature0 = lstm_out[:,-1,:] # (10, 1920)
        lstm_out1, _ = self.lstm(x[:,::2,:])
        feature1 = lstm_out1[:,-1,:]
        lstm_out2, _ = self.lstm(x[:,::4,:])
        feature2 = lstm_out2[:,-1,:]
        feature = torch.cat([feature0, feature1], 1)
        feature = torch.cat([feature, feature2], 1) # (10,5760)
        feature = feature.permute(1,0)

        head_weight = self.dense2(feature)
        head_weight = self.leaky_relu2(head_weight)
        tail_weight = self.dense3(feature)
        tail_weight = self.leaky_relu3(tail_weight)
        weight = torch.matmul(head_weight, self.all_one.float().permute(1,0)) + \
               torch.matmul(self.all_one.float(), tail_weight.permute(1,0))
        
        weight = self.softmax(weight)
        outputs_proped = torch.matmul(weight, feature)
        outputs_concated = torch.cat([feature, outputs_proped], 0)

        linear_out = self.linear(outputs_concated.permute(1,0))
        return linear_out