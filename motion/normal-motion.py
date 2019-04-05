from __future__ import division
import numpy as np


from pylab import *
from numpy import ma
import cv2
import os
from scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import scipy.io as sio

import tensorflow as tf


def prepare_data_surreal(dataset_dir, time_length):
    train_names={'input':[], 'output':[]}

    print('Reading training files...')

    for path, subdirs, files in os.walk(dataset_dir + "\\train"):
        for name in files:
            if '.mp4' in name:
                train_names['input'].append(os.path.join(path, name))
            if '_seg' in name:
                train_names['output'].append(os.path.join(path, name))


    return train_names



def prepare_data_refresh(dataset_dir, time_length):
    train_names={'input':[], 'output':[]}
    input_names = []
    output_names = []
    print('Reading files...')
    num_files = 0
    num_folders = 0
    for path, subdirs, files in os.walk(dataset_dir + "\\training"):

        if 'keyframe_1' in path and 'keyframe_10' not in path:
                if 'raw_color' in path:
                    train_names['input'].append(path)
                if 'rigidity' in path:
                    train_names['output'].append(path)


    print('Done reading files!')


    return train_names



def distance_transform( vol, mode ='unsigned'):
    eps = 1e-15
    if mode == 'unsigned':
        img_output_3d = ndimage.distance_transform_edt(vol)
        img_output_3d = (img_output_3d - (np.min(img_output_3d))) / (np.max(img_output_3d) - np.min(img_output_3d)+ eps)
    if mode == 'signed':
        img_output_3d = ndimage.distance_transform_edt(vol)
        img_output_3d = (img_output_3d - (np.min(img_output_3d))) / (np.max(img_output_3d) - np.min(img_output_3d)+ eps)
        inside = vol == 0.0
        temp = ndimage.distance_transform_edt(1 - vol)
        temp = (temp - (np.min(temp))) / (np.max(temp) - np.min(temp) + eps)
        img_output_3d = np.where(inside,-temp, img_output_3d)
    elif mode == 'thresh-signed':
        img_output_3d = ndimage.distance_transform_edt(vol)
        inside = vol == 0.0
        temp = ndimage.distance_transform_edt(1 - vol)
        img_output_3d = np.where(inside,np.maximum(-temp,-1000), np.minimum(img_output_3d,1000))
        img_output_3d = (img_output_3d - (np.min(img_output_3d))) / (np.max(img_output_3d) - np.min(img_output_3d)+ eps)
        # np.savetxt('C:\\Users\\ajamgard.1\\Desktop\\TemporalPose\\tx.txt',img_output_3d[0,:,:], delimiter=',')
        img_output_3d = img_output_3d * 2.0 - 1.0
    return img_output_3d


def compute_normals(img):

	h = [-0.5,0.,0.5]
	w = [-0.5,0.,0.5]
	t = [-0.5,0.,0.5]
	tt, hh, ww = np.meshgrid(t,h,w,indexing='ij')
	tt = np.expand_dims(tt, axis = 3)
	tt = np.expand_dims(tt, axis = 4)
	tt = tf.constant(tt, dtype=tf.float32)
	hh = np.expand_dims(hh, axis = 3)
	hh = np.expand_dims(hh, axis = 4)
	hh = tf.constant(hh, dtype=tf.float32)
	ww = np.expand_dims(ww, axis = 3)
	ww = np.expand_dims(ww, axis = 4)
	ww = tf.constant(ww, dtype=tf.float32)


	dt = tf.nn.conv3d(img, tt, strides=[1,1,1,1,1], padding='SAME', name ='convT')
	dh = tf.nn.conv3d(img, hh, strides=[1,1,1,1,1], padding='SAME', name ='convH')
	dw = tf.nn.conv3d(img, ww, strides=[1,1,1,1,1], padding='SAME', name ='convW')
	normals = tf.concat([dt, dh, dw],4)
	norm = tf.norm(normals, axis = 4, keepdims = True)
	#
	dt /= (norm + 1e-12)
	dh /= (norm + 1e-12)
	dw /= (norm + 1e-12)
	#
	normals = tf.concat([dt, dh, dw],4)
	return normals

class Stacker:
    def __init__(self, info, time_length):
        self.info = info
        self.time_length =time_length

    def get_refresh(self, frame_number, scale = 6):
        """ Genereate Distance Transform from joints
            by creating thick skeletons
        """
        """ Genereate Distance Transform from joints
            by creating thick skeletons
        """
        size = 128 * 4
        img_input_3d = np.zeros((self.time_length,size, size,3), dtype = np.uint8)
        img_output_3d_inter = np.zeros((self.time_length * scale,size, size), dtype = np.uint8)
        img_output_3d = np.zeros((self.time_length,size, size), dtype = np.uint8)
        i = 0


        inp = self.info['input'][frame_number]
        out = self.info['output'][frame_number]
        while i < self.time_length:

            frame = cv2.imread(os.path.join(inp, '%06d.png'%(i)))
            # read uint16 and get R channel ->opencv channels BGR
            temp = cv2.imread(os.path.join(out, '%06d.png'%(i)))[:,:,2]
            # print(temp.shape)
            # print(cv2.resize(frame, (size,size)).shape)

            img_input_3d[i ,:,:,:] = cv2.resize(frame, (size,size))

            #inside == 0, outside == 1
            temp = cv2.resize(temp, (size,size))
            f = temp == 255

            temp = np.where(f, 0, 1)

            img_output_3d_inter[i * scale,:,:] = temp


            i +=1

        img_output_3d_inter =  distance_transform(img_output_3d_inter, mode ='thresh-signed')
        img_output_3d = img_output_3d_inter[0::scale,:,:]
        # for i in range(0,self.time_length):
            # a = img_output_3d[i,:,:]
            # a = (a - 1.0) / -2.0 * 255.0
            # im_color = cv2.applyColorMap(np.uint8(a), cv2.COLORMAP_JET)
            # numpy_horizontal = np.hstack((im_color, img_input_3d[i,:,:,:]))
            # cv2.imshow('s',numpy_horizontal )
            # cv2.waitKey(0)

        return img_output_3d, img_input_3d


    def get_surreal(self,  frame_number, scale = 6):
        """ Genereate Distance Transform from joints
            by creating thick skeletons
        """
        size = 128 * 4
        img_input_3d = np.zeros((self.time_length,size, size,3), dtype = np.uint8)
        img_output_3d_inter = np.zeros((self.time_length * scale,size, size), dtype = np.uint8)
        img_output_3d = np.zeros((self.time_length,size, size), dtype = np.uint8)



        cap = cv2.VideoCapture(self.info['input'][frame_number])


        output_mat = sio.loadmat(self.info['output'][frame_number])
        i = 0


        while i < self.time_length:
            ret, frame = cap.read()
            img_input_3d[i,:,:,:] = cv2.resize(frame, (size,size))
            temp =  output_mat['segm_{}'.format(i+1)]
            #inside == 0, outside == 1
            temp = cv2.resize(temp, (size,size))
            temp = np.where(temp ==0, 1 - temp, 0)
            img_output_3d_inter[i*scale,:,:] = temp
            i += 1
        cap.release()





        img_output_3d_inter =  distance_transform(img_output_3d_inter, mode ='thresh-signed')
        img_output_3d = img_output_3d_inter[0::scale,:,:]


        return img_output_3d, img_input_3d



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)


time_length = 50
base_dir = os.getcwd()
train_lab = prepare_data_surreal(os.path.join(base_dir , "..\\Data\\SURREAL"), time_length)
stacks_train = Stacker(train_lab, time_length)
img_output, img_input = stacks_train.get_surreal(100, 3)

input_image_batch = np.expand_dims(img_output, axis = 4)
img = tf.placeholder(tf.float32, [None, time_length, 512, 512, 1])
normals = compute_normals(img)

with sess.as_default():
	norm = sess.run(normals, feed_dict = {img : [input_image_batch]})

	for i in range(time_length):
		a = img_output[i,:,:]
		a = (a - 1.0) / -2.0 * 255.0
		im_color = cv2.applyColorMap(np.uint8(a), cv2.COLORMAP_JET)
		# numpy_horizontal = np.hstack((img_input[i,:,:,:], im_color,0.5*(norm[0,i,:,:,:] + 1)))
		cv2.imshow('image',img_input[i,:,:,:] )
		cv2.imshow('SDF',im_color )
		cv2.imshow('Normal',0.5*(norm[0,i,:,:,:] + 1 ))

		cv2.imwrite("C:\\Users\\ajamgard.1\\Box\\Publications\\Materials\\sdf-normals-img\\img%d.png"%i,img_input[i,:,:,:])
		cv2.imwrite("C:\\Users\\ajamgard.1\\Box\\Publications\\Materials\\sdf-normals-img\\sdf%d.png"%i, im_color)
		cv2.imwrite("C:\\Users\\ajamgard.1\\Box\\Publications\\Materials\\sdf-normals-img\\normal%d.png"%i, 0.5*(norm[0,i,:,:,:] + 1 )*255.0)

		cv2.waitKey(0)
