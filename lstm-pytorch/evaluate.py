import numpy as np
import os
from data_preprocess import normalize, meanfilter, datatoseq
from model_lstm import VanillaLSTM, TALSTM, D2TALSTM
import configparser

config = configparser.ConfigParser()
path = os.path.split(os.path.realpath(__file__))[0] + '/model.conf'
config.read(path)
test_save_path = str(config.get('LstmSection', 'test_save_path'))
val_save_path = str(config.get('LstmSection', 'val_save_path'))

