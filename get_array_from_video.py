import cv2
import numpy as np

video_full_path="/home/lby/fabric/破洞_artidef1.mp4"
cap = cv2.VideoCapture(video_full_path)
print(cap.isOpened())
cap.set(cv2.CAP_PROP_POS_FRAMES,7650)  #275, 500, 2875
frame_count = 1
success = True
frame_array = np.zeros((10,1920,1200)) # , 1080,1440, 720,1280
while(success):
    success, frame = cap.read()
    # print('Read a new frame: ', success )

    params = []
    #params.append(cv.CV_IMWRITE_PXM_BINARY) 
    params.append(1)
    frame = frame#[490:1080,90:1440]
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if(frame_count>0):
        frame_array[:,:,frame_count-1] = frame[0:10,:]
        # cv2.imwrite("/home/lby/fabric/pborke_sample" + "_%d.jpg" % frame_count, frame, params) 
        # cv2.imwrite("/home/lby/fabric/video" + "_%d.jpg" % frame_count, frame, params) 
    frame_count = frame_count + 1
      # read the 1000th frame
    if(frame_count>1200):
        break
cap.release()
np.save('/home/lby/fabric/test1200_hole.npy', frame_array)
