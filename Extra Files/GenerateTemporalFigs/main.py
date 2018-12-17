import cv2
import numpy as np
import os

input_names = []
#### Images
# base = '/Users/ajamgard.1/Desktop/SandBox/Segmentation/Data/leftImg8bit_demoVideo/leftImg8bit/demoVideo/stuttgart_01/'
#
# for file in os.listdir(base):
#     if '.png' in file:
#         cwd = os.getcwd()
#         input_names.append( base + file)
# input_names = np.sort(input_names)
# img = cv2.imread(input_names[0])
#
#
# xt_slice  = []
# yt_slice  = []
# print('Finished reading image names.')
#
# for i in range(400):
#     if i%20 == 0: print(i)
#     img = cv2.imread(input_names[i])
#     img = cv2.resize(img,None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)
#
#     xt_slice.append(img[:,120,:].reshape((1,img.shape[0],3)))
#     yt_slice.append(img[120,:,:].reshape((1,img.shape[1],3)))
    # img1 = cv2.line(img, (0,120), (500, 120), (255,0,0),3)[:]
    # cv2.imshow('sd', img1)
    # if cv2.waitKey(1) &  0xFF == ord('q'):
    #         break
###### Video
cap = cv2.VideoCapture('Nima.MOV')
xt_slice  = []#np.zeros((300,img.shape[0], 3), dtype = np.uint8)
yt_slice  = []#np.zeros((300,img.shape[1],3) , dtype = np.uint8)
i = 0
while (cap.isOpened() and i < 400):
    i+=1
    ret, frame = cap.read()
    if not ret :
        break
    frame = cv2.resize(frame,None,fx=0.267, fy=0.267, interpolation = cv2.INTER_CUBIC)
    rows = frame.shape[0]
    cols = frame.shape[1]

    M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)

    frame = cv2.warpAffine(frame,M,(cols,rows))

    cv2.imshow('xt', frame)
    if cv2.waitKey(1) &  0xFF == ord('q'):
        break

    xt_slice.append(frame[:,133,:].reshape((1,frame.shape[0],3)))
    yt_slice.append(frame[133,:,:].reshape((1,frame.shape[1],3)))
#
xt_slice = np.asarray(xt_slice)
yt_slice = np.asarray(yt_slice)
xt_slice = xt_slice.reshape(xt_slice.shape[0],xt_slice.shape[2],xt_slice.shape[3])
yt_slice = yt_slice.reshape(yt_slice.shape[0],yt_slice.shape[2],yt_slice.shape[3])
# cap.release()
cv2.imshow('xt',xt_slice)
cv2.imshow('yt', yt_slice)
cv2.imwrite('yt1.png', yt_slice)
cv2.waitKey(0)
