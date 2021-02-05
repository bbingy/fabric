import numpy as np
import os
import cv2
from skimage import measure
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
# seq_length = int(config.get('LstmSection', 'seq_length'))
sum_length = 50
sum_width = 15
isblack = 50
seq_length = 200
draw_ID = 1781

def get_threshold(pred_dir, gt_dir, sum_length, sum_width):
    pred_data = np.load(pred_dir) # (3400,1920)
    gt_data = np.load(gt_dir)
    pred_data[np.where(pred_data>np.max(gt_data))] = np.max(gt_data)
    pred_data[np.where(pred_data<np.min(gt_data))] = np.min(gt_data)
    dist = (pred_data-gt_data)
    full_length, width = pred_data.shape
    sum_dist = cv2.blur(dist, (sum_width, sum_length))
    # sum_dist = np.empty([full_length, width])
    # for i in range(full_length):
    #     if i+sum_length>full_length:
    #         sum_dist[i,:] = np.sum(dist[i:,:], axis=0)  # might cause fault
    #     else:
    #         sum_dist[i,:] = np.sum(dist[i:i+sum_length,:], axis=0)
    max_dist = np.max(sum_dist)
    # draw_pred(gt_data[2400:,draw_ID], pred_data[2400:,draw_ID], sum_dist[2400:,draw_ID]/1)
    return max_dist

def get_defect_pixel(pred_hole_dir, gt_hole_dir, thr, sum_length, sum_width):
    pred_data = np.load(pred_hole_dir)
    gt_data = np.load(gt_hole_dir)
    pred_data[np.where(pred_data>np.max(gt_data))] = np.max(gt_data)
    pred_data[np.where(pred_data<np.min(gt_data))] = np.min(gt_data)
    dist = (pred_data-gt_data)
    full_length, width = pred_data.shape
    sum_dist = cv2.blur(dist, (sum_width, sum_length))  #[1000, 1920]
    # sum_dist = np.empty([full_length, width])
    # for i in range(full_length):
    #     if i+sum_length>full_length:
    #         sum_dist[i,:] = np.sum(dist[i:,:], axis=0)  # might cause fault
    #     else:
    #         sum_dist[i,:] = np.sum(dist[i:i+sum_length,:], axis=0)
    max_dist = np.max(sum_dist)
    # draw_on_frame(279+7650, draw_ID)
    # draw_pred(gt_data[:,draw_ID], pred_data[:,draw_ID], sum_dist[:,draw_ID]/1)
    defect_pixel = np.where(sum_dist>thr)
    return sum_dist, defect_pixel

def get_defect(sum_dist, defect_pixel):
    # defect_pixel = sorted(defect_pixel,key=lambda defect_pixel:defect_pixel[0])
    # num_left = len(defect_pixel[0])  #165356
    # defect_list = []
    # while(num_left>0):
    #     defect_loc_f = defect_pixel[0][0]
    #     defect_loc_x = defect_pixel[1][0]
    mask = np.zeros(sum_dist.shape)
    mask[defect_pixel] = 1
    label, num = measure.label(mask, neighbors=8, background=0, return_num=True)
    return label, num

def evaluate(pred_defect_loc, gt_label_dir, seq_length, isblack):
    gt_label = np.load(gt_label_dir)  # (60,1920,1200)
    num_pred = len(pred_defect_loc[0])
    num_defect = gt_label.shape[0]
    TP_array = np.zeros([num_defect])
    FP = 0
    for i in range(num_pred):
        FP_flag = 1
        for j in range(num_defect):
            if np.any(gt_label[j, pred_defect_loc[1][i], pred_defect_loc[0][i]+seq_length:pred_defect_loc[0][i]+seq_length+sum_length]<isblack):
                TP_array[j] += 1
                FP_flag = 0
        if FP_flag==1:
            FP += 1
    # compute TP and FN
    FN_local = np.sum(TP_array==0)
    TP_local = num_defect-FN_local
    TP = np.sum(TP_array)
    precision = TP/num_pred
    recall = TP/np.sum(gt_label<isblack)
    F1 = 2*precision*recall/(precision+recall)
    # 59, 58444, 1, 190222
    return TP_local, precision, recall, F1

def draw_pred(gt, pred, sum_dist):
    gt_label_dir = '/home/lby/fabric/test1200_label.npy'
    gt_label = np.load(gt_label_dir)

    plt.figure(figsize=(30,10))
    plt.plot(gt)
    plt.plot(pred, color='red')
    plt.plot(sum_dist, color='black')
    # plt.plot(gt_label[34,draw_ID,200:]/255, color='yellow')

    plt.legend(['gt', 'pred', 'error'], loc='best') # , 'label'
    plt.show()
    plt.savefig('/home/lby/fabric/lstm-pytorch/pred.jpg')

def draw_on_frame(frame_id, loc_x):
    video_full_path="/home/lby/fabric/破洞_artidef1.mp4"
    # video_full_path="/home/lby/fabric/label_mask/04.mp4"
    cap = cv2.VideoCapture(video_full_path)
    print(cap.isOpened())
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id)
    _, frame = cap.read()
    cv2.rectangle(frame, (loc_x-5, 30), (loc_x+5, 40), (0, 0, 255), thickness=-1)
    cv2.imwrite("/home/lby/fabric/pred_frame.jpg", frame) 

def draw_label(pred_label, gt_label_dir, isblack):
    defect_area = np.where(pred_label>0)
    pred_label[defect_area] = 255
    cv2.imwrite('/home/lby/fabric/pred_label.jpg', pred_label)
    gt_label = np.load(gt_label_dir)
    defect_area = np.where(gt_label<isblack)
    full_gt_label = np.zeros([gt_label.shape[1], gt_label.shape[2]])
    full_gt_label[defect_area[1], defect_area[2]] = 255
    cv2.imwrite('/home/lby/fabric/gt_label.jpg', full_gt_label)

if __name__ == '__main__':
    pred_dir = '/home/lby/fabric/lstm-pytorch/test_seq/d2talstm200.npy'
    gt_dir = '/home/lby/fabric/test3600_gt_200.npy'
    pred_hole_dir = '/home/lby/fabric/lstm-pytorch/test_seq/d2talstm200_hole.npy'
    gt_hole_dir = '/home/lby/fabric/test1200_gt_200.npy'
    gt_label_dir = '/home/lby/fabric/test1200_label.npy'
    thr = get_threshold(pred_dir, gt_dir, sum_length, sum_width)
    sum_dist, pred_defect_pixel = get_defect_pixel(pred_hole_dir, gt_hole_dir, thr, sum_length, sum_width)
    pred_label, pred_num = get_defect(sum_dist, pred_defect_pixel)
    draw_label(pred_label, gt_label_dir, isblack)
    TP_local, precision, recall, F1 = evaluate(pred_defect_pixel, gt_label_dir, seq_length, isblack)