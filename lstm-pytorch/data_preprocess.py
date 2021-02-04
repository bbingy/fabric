import numpy as np
import cv2
import configparser
import os
import pickle

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

def datatoseq(data, seq_length):
    batch_size, data_length = data.shape
    num_seq = data_length-seq_length
    X = np.empty([num_seq, batch_size, seq_length])
    Y = np.empty([num_seq, batch_size])
    for i in range(num_seq):
        X[i,:,:] = data[:,i:i+seq_length]
        Y[i,:] = data[:,i+seq_length]
    return X, Y

def data_prepare(data_dir, seq_length):
    data = np.load(data_dir)
    data = normalize(data)
    data = meanfilter(data)
    X, Y = datatoseq(data, seq_length)
    return X, Y

def get_test_label(data_dir):
    files = os.listdir(data_dir)
    files.sort()
    num_defect = len(files)
    label_list = []
    label_list = np.empty([num_defect,1920,1200])
    file_id = 0
    for file in files:
        file_dir = data_dir+file
        print(file_dir)
        cap = cv2.VideoCapture(file_dir)
        cap.set(cv2.CAP_PROP_POS_FRAMES,7650)
        # label_array = np.empty([1920,1200])
        frame_count = 1
        success = True
        while(success):
            success, frame = cap.read()

            params = []
            params.append(1)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            label_list[file_id,:,frame_count-1] = frame[4,:]
            # label_array[:,frame_count-1] = frame[4,:]
            frame_count = frame_count + 1
            if(frame_count>1200):
                break
        cap.release()
        file_id = file_id+1
        # label_list.append(label_array)
    print(label_list.shape)
    np.save('/home/lby/fabric/test1200_label.npy', label_list)
    # print(len(label_list))
    # with open("/home/lby/fabric/test1200_label.txt", "wb") as fp:
    #     pickle.dump(label_list, fp)

if __name__ == '__main__':
    data_dir = '/home/lby/fabric/test3600_nohole.npy'
    label_dir = '/home/lby/fabric/label_mask/'
    get_test_label(label_dir)
    # data = np.load(data_dir)
    # data = normalize(data)
    # print(data.shape)
    # data = meanfilter(data)
    # print(data.shape)
    # X, Y = datatoseq(data, seq_length)
    # np.save('/home/lby/fabric/test3600_gt.npy', Y)