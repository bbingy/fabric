import numpy as np
import cv2
import configparser
import os

config = configparser.ConfigParser()
path = os.path.split(os.path.realpath(__file__))[0] + '/model.conf'
config.read(path)
seq_length = int(config.get('LstmSection', 'seq_length'))
ker_size = int(config.get('LstmSection', 'ker_size'))

def normalize(data):
    data_mean = np.load('/home/lby/fabric/nohole_mean.npy')
    data_std = np.load('/home/lby/fabric/nohole_std.npy')
    mean_loc = data_mean[0:10, :]
    std_loc = data_std[0:10, :]
    data = (data-np.expand_dims(mean_loc, axis=2))/np.expand_dims(std_loc, axis=2)
    return data

def meanfilter(data):
    _, h, num_f = data.shape
    data_blur = np.empty([h,num_f])
    for i in range(num_f):
        img_blur = cv2.blur(data[:,:,i], (ker_size, ker_size))
        data_blur[:,i] = img_blur[4,:]
    return data_blur

def datatoseq(data):
    batch_size, data_length = data.shape
    num_seq = data_length-seq_length
    X = np.empty([num_seq, batch_size, seq_length])
    Y = np.empty([num_seq, batch_size])
    for i in range(num_seq):
        X[i,:,:] = data[:,i:i+seq_length]
        Y[i,:] = data[:,i+seq_length]
    return X, Y

def data_prepare(data_dir):
    data = np.load(data_dir)
    data = normalize(data)
    data = meanfilter(data)
    X, Y = datatoseq(data)
    return X, Y

if __name__ == '__main__':
    data_dir = '/home/lby/fabric/test1200_hole.npy'
    # ker_size = 5
    # seq_length = 100
    data = np.load(data_dir)
    data = normalize(data)
    print(data.shape)
    data = meanfilter(data)
    print(data.shape)
    X, Y = datatoseq(data)