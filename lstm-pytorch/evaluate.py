import numpy as np
import os
import cv2
from data_preprocess import normalize, meanfilter, datatoseq
# from model_lstm import VanillaLSTM, TALSTM, D2TALSTM
import configparser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
path = os.path.split(os.path.realpath(__file__))[0] + '/model.conf'
config.read(path)
test_save_path = str(config.get('LstmSection', 'test_save_path'))
val_save_path = str(config.get('LstmSection', 'val_save_path'))
seq_length = int(config.get('LstmSection', 'seq_length'))
sum_length = 10
min_width = 10
draw_ID = 71

def get_threshold(pred_dir, gt_dir):
    pred_data = np.load(pred_dir) # (3400,1920)
    gt_data = np.load(gt_dir)
    dist = (pred_data-gt_data)
    full_length, width = pred_data.shape
    sum_dist = cv2.blur(dist, (min_width, sum_length))
    # sum_dist = np.empty([full_length, width])
    # for i in range(full_length):
    #     if i+sum_length>full_length:
    #         sum_dist[i,:] = np.sum(dist[i:,:], axis=0)  # might cause fault
    #     else:
    #         sum_dist[i,:] = np.sum(dist[i:i+sum_length,:], axis=0)
    max_dist = np.max(sum_dist)
    draw_pred(gt_data[2400:,draw_ID], pred_data[2400:,draw_ID], sum_dist[2400:,draw_ID]/1)
    return max_dist

def get_defect_pixel(pred_hole_dir, gt_hole_dir, thr):
    pred_data = np.load(pred_hole_dir)
    gt_data = np.load(gt_hole_dir)
    dist = (pred_data-gt_data)
    full_length, width = pred_data.shape
    sum_dist = cv2.blur(dist, (min_width, sum_length))
    # sum_dist = np.empty([full_length, width])
    # for i in range(full_length):
    #     if i+sum_length>full_length:
    #         sum_dist[i,:] = np.sum(dist[i:,:], axis=0)  # might cause fault
    #     else:
    #         sum_dist[i,:] = np.sum(dist[i:i+sum_length,:], axis=0)
    max_dist = np.max(sum_dist)
    draw_pred(gt_data[:,draw_ID], pred_data[:,draw_ID], sum_dist[:,draw_ID]/1)
    defect_pixel = np.where(sum_dist>thr)
    return defect_pixel

# def get_defect_loc(defect_pixel):

def evaluate(pred_defect_loc, gt_label_dir):
    gt_label = np.load(gt_label_dir)  # (60,1920,1200)
    num_pred = len(pred_defect_loc[0])
    num_defect = gt_label.shape[0]
    TP_array = np.zeros([num_defect])
    FP = 0
    for i in range(num_pred):
        FP_flag = 1
        for j in range(num_defect):
            if np.any(gt_label[j, pred_defect_loc[1][i], pred_defect_loc[0][i]+seq_length:pred_defect_loc[0][i]+seq_length+sum_length]<20):
                TP_array[j] += 1
                FP_flag = 0
        if FP_flag==1:
            FP += 1
    # compute TP and FN
    FN = np.sum(TP_array==0)
    TP = num_defect-FN
    # 59, 58444, 1, 190222
    return TP, FP, FN

def draw_pred(gt, pred, sum_dist):
    gt_label_dir = '/home/lby/fabric/test1200_label.npy'
    gt_label = np.load(gt_label_dir)

    plt.figure(figsize=(30,10))
    plt.plot(gt)
    plt.plot(pred, color='red')
    # plt.plot(sum_dist, color='black')
    plt.plot(gt_label[34,draw_ID,200:]/255, color='yellow')

    plt.legend(['gt', 'pred', 'label'], loc='best') # , 'error'
    plt.show()
    plt.savefig('/home/lby/fabric/lstm-pytorch/pred.jpg')

if __name__ == '__main__':
    pred_dir = '/home/lby/fabric/lstm-pytorch/test_seq/d2talstm30.npy'
    gt_dir = '/home/lby/fabric/test3600_gt_30.npy'
    pred_hole_dir = '/home/lby/fabric/lstm-pytorch/test_seq/d2talstm30_hole.npy'
    gt_hole_dir = '/home/lby/fabric/test1200_gt_30.npy'
    gt_label_dir = '/home/lby/fabric/test1200_label.npy'
    thr = get_threshold(pred_dir, gt_dir)
    pred_defect_loc = get_defect_pixel(pred_hole_dir, gt_hole_dir, thr)
    TP, FP, FN = evaluate(pred_defect_loc, gt_label_dir)