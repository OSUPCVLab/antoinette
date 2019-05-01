"""
Temporal Pose
Common utility functions and classes.

Copyright (c) 2018 PCVLab & ADL
Licensed under the MIT License (see LICENSE for details)
Written by Nima A. Gard
"""
from __future__ import division
import numpy as np
import cv2
import os
from itertools import islice
import datetime
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
from scipy import ndimage
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import random
import ntpath
from skimage import measure
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import PIL

from matplotlib import colors as mcolors
from utils.vtk_utils import *

# import tf
def prepare_video(dataset_dir, time_length, mode = 'TRAIN'):
    test_names={'input':[], 'output':[]}
    for path, subdirs, files in os.walk(dataset_dir):
        for name in files:
            if '.MOV' in name:
                test_names['input'].append(os.path.join(path, name))

    return test_names

def prepare_data(dataset_dir, time_length, mode = 'TRAIN'):
    train_names={'input':[], 'output':[]}
    val_names={'input':[], 'output':[]}
    test_names={'input':[], 'output':[]}
    print('Reading training files...')

    for path, subdirs, files in os.walk(dataset_dir + "\\train"):
        for name in files:
            if '.mp4' in name:
                train_names['input'].append(os.path.join(path, name))
            if '_seg' in name:
                train_names['output'].append(os.path.join(path, name))
    print('Reading validation files...')
    for path, subdirs, files in os.walk(dataset_dir + "\\train"):
        for name in files:
            if '.mp4' in name:
                val_names['input'].append(os.path.join(path, name))
            if '_seg' in name:
                val_names['output'].append(os.path.join(path, name))
    print('Reading test files...')
    for path, subdirs, files in os.walk(dataset_dir + "\\test"):
        for name in files:
            if '.mp4' in name:
                test_names['input'].append(os.path.join(path, name))
            if '_seg' in name:
                test_names['output'].append(os.path.join(path, name))
    train_names['input'] = np.sort(train_names['input'])
    train_names['output'] = np.sort(train_names['output'])
    val_names['input'] = np.sort(val_names['input'])
    val_names['output'] =np.sort( val_names['output'])
    test_names['input'] = np.sort(test_names['input'])
    test_names['output'] = np.sort(test_names['output'])

    # # choose 2000 sample
    train_ind = random.sample(range(0,len(train_names['input'])),200)
    train_names['input'] = train_names['input'][train_ind]
    train_names['output'] = train_names['output'][train_ind]
    print('Done reading files!')
    print('Cleaning files..')
    if mode == 'TRAIN':
        train_names = remove_frame_deficient(train_names,time_length)
    # val_names = remove_frame_deficient(val_names,time_length)
    # test_names = remove_frame_deficient(test_names,time_length)
    print('Done cleaning files!')

    return train_names,val_names,test_names

def prepare_data_synthia(dataset_dir, time_length):
    train_names={'input':[], 'output':[]}
    val_names={'input':[], 'output':[]}
    test_names={'input':[], 'output':[]}
    input_names = []
    output_names = []
    print('Reading files...')
    num_files = 0
    num_folders = 0
    for path, subdirs, files in os.walk(dataset_dir + "\\RGB\\Stereo_Right"):
        for name in files:
            input_names.append(os.path.join(path, name))
        num_files += len(files)
        num_folders += len(subdirs)
    # input_names = input_names[:-time_length]
    for path, subdirs, files in os.walk(dataset_dir + "\\GT\\LABELS\\Stereo_Right"):
        for name in files:
            output_names.append(os.path.join(path, name))
    # output_names = output_names[:-time_length]

    train_ind = []
    val_ind = []
    test_ind = []


    ids = np.arange(0,num_files,time_length, dtype = np.int32)
    train_ind = np.random.choice(ids,int(len(ids)*0.70),replace=False)
    ids = np.setdiff1d(ids,train_ind)
    val_ind = np.random.choice(ids,int(len(ids)*0.5),replace=False)
    test_ind = np.setdiff1d(ids,val_ind)

    input_names = np.asarray(input_names)
    output_names = np.asarray(output_names)

    train_names['input'] = np.sort(input_names[ids])
    train_names['output'] = np.sort(output_names[ids])
    val_names['input'] = np.sort(input_names[val_ind])
    val_names['output'] =np.sort( output_names[val_ind])
    test_names['input'] = np.sort(input_names[test_ind])
    test_names['output'] = np.sort(output_names[test_ind])

    print('Done reading files!')


    return train_names,val_names,test_names


def prepare_data_refresh(dataset_dir, time_length):
    train_names={'input':[], 'output':[]}
    val_names={'input':[], 'output':[]}
    test_names={'input':[], 'output':[]}
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

    for path, subdirs, files in os.walk(dataset_dir + "\\val"):
        if 'keyframe_1' in path and 'keyframe_10' not in path:
                if 'raw_color' in path:
                    val_names['input'].append(path)
                if 'rigidity' in path:
                    val_names['output'].append(path)

    for path, subdirs, files in os.walk(dataset_dir + "\\test"):
        if 'keyframe_1' in path and 'keyframe_10' not in path:
                if 'raw_color' in path:
                    test_names['input'].append(path)
                if 'rigidity' in path:
                    test_names['output'].append(path)


    print('Done reading files!')


    return train_names,val_names,test_names


def prepare_data_posetrack(dataset_dir, time_length):
    train_names={'input':[], 'output':[]}
    val_names={'input':[], 'output':[]}
    test_names={'input':[], 'output':[]}
    input_names = []
    output_names = []
    print('Reading files...')
    num_files = 0
    num_folders = 0
    for path, subdirs, files in os.walk(dataset_dir + "\\train"):
        for sd in subdirs:
            train_names['input'].append(os.path.join(path, sd))

    for path, subdirs, files in os.walk(dataset_dir + "\\val"):
        for sd in subdirs:
            val_names['input'].append(os.path.join(path, sd))


    for path, subdirs, files in os.walk(dataset_dir + "\\test"):
        for sd in subdirs:
            test_names['input'].append(os.path.join(path, sd))


    print('Done reading files!')


    return train_names,val_names,test_names

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
        img_output_3d = np.where(inside,np.maximum(-temp,-200), np.minimum(img_output_3d,200))
        img_output_3d = (img_output_3d - (np.min(img_output_3d))) / (np.max(img_output_3d) - np.min(img_output_3d)+ eps)
        # np.savetxt('C:\\Users\\ajamgard.1\\Desktop\\TemporalPose\\tx.txt',img_output_3d[0,:,:], delimiter=',')
        img_output_3d = img_output_3d * 2.0 - 1.0
    return img_output_3d

class Stacker:
    def __init__(self, info, time_length):
        self.info = info
        self.time_length =time_length


    def get_data_synthia(self,  frame_number):
        """ Genereate Distance Transform from joints
            by creating thick skeletons
        """
        size = 128
        img_input_3d = np.zeros((self.time_length,size, size,3), dtype = np.uint8)
        img_output_3d = np.zeros((self.time_length,size, size), dtype = np.uint8)


        i = 0

        head_in, _  = ntpath.split(self.info['input'][frame_number])
        head_out, _  = ntpath.split(self.info['output'][frame_number])
        while i < self.time_length:
            frame = cv2.imread(os.path.join(head_in, '%06d.png'%(i + frame_number)))
            # read uint16 and get R channel ->opencv channels BGR
            temp = cv2.imread(os.path.join(head_out, '%06d.png'%(i + frame_number)),cv2.IMREAD_UNCHANGED)[:,:,2]

            img_input_3d[i,:,:,:] = cv2.resize(frame, (size,size))
            #inside == 0, outside == 1
            temp = cv2.resize(temp, (size,size))
            f = temp == (8 or 10)

            temp = np.where(f, 0, 1)
            img_output_3d[i,:,:] = temp
            i += 1

        img_output_3d =  distance_transform(img_output_3d, mode ='thresh-signed')

        # cv2.imshow('s',img_output_3d[0,:,:])
        # Visualizer_3D().visualize_3d_volume(img_output_3d)
        # cv2.waitKey(0)

        return img_output_3d, img_input_3d




    def get_data(self,  frame_number):
        """ Genereate Distance Transform from joints
            by creating thick skeletons
        """
        size = 128
        img_input_3d = np.zeros((self.time_length,size, size,3), dtype = np.uint8)
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
            img_output_3d[i,:,:] = temp
            i += 1
        cap.release()





        img_output_3d =  distance_transform(img_output_3d, mode ='thresh-signed')
        # for i in range(16):
            # img_output_3d[i,:,:] = cv2.normalize(img_output_3d[i,:,:],  0, 255, cv2.NORM_MINMAX)
        # cv2.imshow('s',img_output_3d[0,:,:])
        # Visualizer_3D().visualize_3d_volume(img_output_3d)
        #
        # cv2.waitKey(0)
        return img_output_3d, img_input_3d


    def get_video(self,  frame_number):
        """ Genereate Distance Transform from joints
            by creating thick skeletons
        """
        size = 128
        img_input_3d = np.zeros((self.time_length,size, size,3), dtype = np.uint8)
        cap = cv2.VideoCapture(self.info['input'][0])

        i = 0
        ii = 0
        while i < self.time_length:
            ret, frame = cap.read()
            h, w = frame.shape[:2]
            if ii >= frame_number*(self.time_length-1):
                # img_input_3d[i,:,:,:] = cv2.resize(frame, (size,size))
                img_input_3d[i,:,:,:] = frame[h//2 - size//2:h//2 + size//2,w//2 - size//2:w//2 + size//2,]
                i += 1
            ii += 1
        cap.release()

        return [], img_input_3d


    def get_video_sq(self,  frame_number):
        """ Genereate Distance Transform from joints
            by creating thick skeletons
        """


# Perform the clockwise rotation holding at the center
# 90 degrees


        size = 128
        img_input_3d = np.zeros((self.time_length,size, size,3), dtype = np.uint8)
        cap = cv2.VideoCapture(self.info['input'][0])

        i = 0
        ii = 0
        while i < self.time_length:
            ret, frame = cap.read()
            (h, w) = frame.shape[:2]
            # calculate the center of the image
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, -90, 1)
            frame = cv2.warpAffine(frame, M, (w, h))

            if ii >= frame_number*(self.time_length-1):
                img_input_3d[i,:,:,:] = cv2.resize(frame[:,120:440,:], (size,size))
                i += 1
            ii += 1
        cap.release()

        return [], img_input_3d



    def get_refresh(self,  frame_number):
        """ Genereate Distance Transform from joints
            by creating thick skeletons
        """
        size = 128
        img_input_3d = np.zeros((self.time_length,size, size,3), dtype = np.uint8)
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
            img_input_3d[i,:,:,:] = cv2.resize(frame, (size,size))
            #inside == 0, outside == 1
            temp = cv2.resize(temp, (size,size))
            f = temp == 255

            temp = np.where(f, 0, 1)
            # img_output_3d_img[i,:,:] = cv2.cvtColor(img_input_3d[i,:,:,:], cv2.COLOR_BGR2GRAY)/255.0
            img_output_3d[i,:,:] = temp

            i += 1

        img_output_3d =  distance_transform(img_output_3d, mode ='thresh-signed')

        ## Method 1)
        # dx,dy,dz= np.gradient(img_output_3d)
        # gradients = np.stack([dx,dy,dz],3)

        ## Method 2)

        # h = [0.5,0.,-0.5]
        # w = [0.5,0.,-0.5]
        # t = [0.5,0.,-0.5]
        # tt, hh, ww = np.meshgrid(t,h,w,indexing='ij')
        # dt = ndimage.convolve(img_output_3d, tt)
        # dh = ndimage.convolve(img_output_3d, hh)
        # dw = ndimage.convolve(img_output_3d, ww)
        # gradients = np.stack([dt,dh,dw],3)
        # gradients[:,:,:,0] /= (np.linalg.norm(gradients,axis = 3) + 1e-15)
        # gradients[:,:,:,1] /= (np.linalg.norm(gradients,axis = 3) + 1e-15)
        # gradients[:,:,:,2] /= (np.linalg.norm(gradients,axis = 3) + 1e-15)
        # for i in range(16*4):
        #     img_output_3d[i,:,:] = cv2.normalize(img_output_3d[i,:,:],  0, 255, cv2.NORM_MINMAX)
        # cv2.imshow('s',img_output_3d[0,:,:])
        # Visualizer_3D().visualize_3d_volume(img_output_3d)

        # cv2.waitKey(0)
        # img_output_3d = np.stack((img_output_3d_img,img_output_3d_impl),axis = 3)
        return img_output_3d, img_input_3d


    def get_refresh_v2(self, frame_number, scale = 4):
        """ Genereate Distance Transform from joints
            by creating thick skeletons
        """
        size = 128
        img_input_3d = np.zeros((self.time_length,size, size,3), dtype = np.uint8)
        img_output_3d_inter = np.zeros((self.time_length * scale,size, size), dtype = np.uint8)
        img_output_3d = np.zeros((self.time_length,size, size), dtype = np.uint8)
        i = 0


        inp = self.info['input'][frame_number]
        out = self.info['output'][frame_number]

        x, y = random_crop(size)

        while i < self.time_length:

            frame = cv2.imread(os.path.join(inp, '%06d.png'%(i)))
            # read uint16 and get R channel ->opencv channels BGR
            temp = cv2.imread(os.path.join(out, '%06d.png'%(i)))[:,:,2]
            # print(temp.shape)
            # print(cv2.resize(frame, (size,size)).shape)

            # img_input_3d[i ,:,:,:] = cv2.resize(frame, (size,size))
            img_input_3d[i ,:,:,:] = frame[x:size+x, y:size+y,:]

            #inside == 0, outside == 1
            # temp = cv2.resize(temp, (size,size))
            temp = temp[x:x+size, y:y+size]
            f = temp == 255

            temp = np.where(f, 0, 1)

            img_output_3d_inter[i * scale,:,:] = temp


            i +=1

        img_output_3d_inter =  distance_transform(img_output_3d_inter, mode ='thresh-signed')
        img_output_3d = img_output_3d_inter[0::scale,:,:]
        # for i in range(0,self.time_length):
        #     a = img_output_3d[i,:,:]
        #     a = (a - 1.0) / -2.0 * 255.0
        #     im_color = cv2.applyColorMap(np.uint8(a), cv2.COLORMAP_JET)
        #     numpy_horizontal = np.hstack((im_color, img_input_3d[i,:,:,:]))
        #     cv2.imshow('s',numpy_horizontal )
        #     cv2.waitKey(0)
        #
        return img_output_3d, img_input_3d


    def get_refresh_v3(self, frame_number, scale = 4):
        """ Genereate Distance Transform from joints
            by creating thick skeletons
        """
        size = 128
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

            img_input_3d[i ,:,:,:] = cv2.resize(frame[:,:480], (size,size))


            #inside == 0, outside == 1
            temp = temp[:, :480]
            temp = cv2.resize(temp, (size,size))
            f = temp == 255

            temp = np.where(f, 0, 1)

            img_output_3d_inter[i * scale,:,:] = temp


            i +=1

        img_output_3d_inter =  distance_transform(img_output_3d_inter, mode ='thresh-signed')
        img_output_3d = img_output_3d_inter[0::scale,:,:]
        # for i in range(0,self.time_length):
        #     a = img_output_3d[i,:,:]
        #     a = (a - 1.0) / -2.0 * 255.0
        #     im_color = cv2.applyColorMap(np.uint8(a), cv2.COLORMAP_JET)
        #     numpy_horizontal = np.hstack((im_color, img_input_3d[i,:,:,:]))
        #     cv2.imshow('s',numpy_horizontal )
        #     cv2.waitKey(0)
        #
        return img_output_3d, img_input_3d

    def get_posetrack(self,  frame_number):
        """ Genereate Distance Transform from joints
            by creating thick skeletons
        """
        size = 128
        img_input_3d = np.zeros((self.time_length,size, size,3), dtype = np.uint8)
        i = 0
        inp = self.info['input'][frame_number]
        print(frame_number)
        while i < self.time_length:

            frame = cv2.imread(os.path.join(inp, '%06d.jpg'%(i)))
            temp = cv2.resize(frame, dsize = None, fx = 0.25, fy = .25)
            (h,w) = temp.shape[:2]
            # print('dims')
            # print(h//2 - 64, h//2 + 64)
            # print(w//2 - 64, 64 + w//2)
            if (h < 128 or w < 128):
                img_input_3d[i,:,:,:] = cv2.resize(temp, (size,size))
                i+= 1
                continue
            img_input_3d[i,:,:,:] = temp[h//2-64:h//2+64, w//2-64:64+w//2,:]

            i += 1

        return [], img_input_3d

# Use a custom OpenCV function to read the image, instead of the standard
# TensorFlow `tf.read_file()` operation.
def _read_py_function(filename, label):
    size = 128
    image_decoded = np.zeros((self.time_length,size, size,3), dtype = np.uint8)
    label_decoded = np.zeros((self.time_length,size, size), dtype = np.uint8)


    i = 0

    head_in, _  = ntpath.split(filename.decode())
    head_out, _  = ntpath.split(label.decode())
    while i < self.time_length:
        frame = cv2.imread(os.path.join(head_in, '%06d.png'%(i + frame_number)))
        # read uint16 and get R channel ->opencv channels BGR
        temp = cv2.imread(os.path.join(head_out, '%06d.png'%(i + frame_number)),cv2.IMREAD_UNCHANGED)[:,:,2]

        image_decoded[i,:,:,:] = cv2.resize(frame, (size,size))
        #inside == 0, outside == 1
        temp = cv2.resize(temp, (size,size))
        f = temp == (8 or 10)

        temp = np.where(f, 0, 1)
        label_decoded[i,:,:] = temp
        i += 1

    label_decoded =  self.distance_transform(label_decoded, mode ='signed')

    return image_decoded, label_decoded




def remove_frame_deficient(data, time_length):
    # data['input'] = np.sort(data['input'])
    # data['output'] = np.sort(data['output'])
    index = []
    print('         Playing videos to find defective clips...')
    for i in range(len(data['input'])):
        cap = cv2.VideoCapture(data['input'][i])
        if  int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) < time_length:
            index.append(i)
            cap.release()
    print('         Finisehd playing videos.')
    data['input'] = np.delete(data['input'], index)
    data['output'] = np.delete(data['output'], index)

    return data




def window(seq, n=3):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

# Print with time. To console or file
def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)


def transparent_overlay(src, overlay):

	color_contour = [0,255,0]
	color_inside = [255,0,0]
	overlay = overlay[:,:,0]
	# print(np.max(overlay),np.min(overlay))
	mid_bound = (np.max(overlay) - np.min(overlay)) / 2 + np.min(overlay)
	upper_bound = 0.90# mid_bound + 0.01 * (np.max(overlay) - np.min(overlay))#127#
	lower_bound = 0.70 #mid_bound - 0.01 * (np.max(overlay) - np.min(overlay))#120#
	ix_contour = np.logical_and(overlay < upper_bound , overlay >lower_bound)
	ix_inside = overlay < 	lower_bound

	ix_contour = np.stack([ix_contour,ix_contour,ix_contour], axis = 2)
	ix_inside = np.stack([ix_inside,ix_inside,ix_inside], axis = 2)

	alpha_contour = np.where(ix_contour, [1.0,1.0,1.0], [0.0,0.0,0.0])
	alpha_inside = np.where(ix_inside,[0.5, 0.5,0.5],[0.0,0.0,0.0])


	alpha = alpha_contour + alpha_inside


	ov_contour = np.where(ix_contour, color_contour * alpha, [0.0,0.0,0.0])
	ov_inside = np.where(ix_inside, color_inside * alpha, [0.0,0.0,0.0])
	channels = ov_contour + ov_inside

	idx = alpha  == 0.0
	src = np.where(idx,src ,  channels + (1.0 - alpha) * src )



	ax.axis('image')
	ax.set_xticks([])
	ax.set_yticks([])
	plt.show()

	return np.uint8(src)


def interpolated_contour(src, overlay, lab):

    bound = 0.90

    color = 'skyblue'
    color_inside = mcolors.to_rgba(color)[:3]
    color_inside = [int(c * 255.0) for c in color_inside][::-1]
    # color_inside = [1.0, 0.7529411764705882, 0.796078431372549]#[135,206,235]
    overlay = overlay[:,:,0]
    ix_inside = overlay < 	bound

    # extract regions
    binary = overlay< bound

    cleared = clear_border(binary)
    # ax.imshow(np.uint8(cleared))
    # plt.show()
    label_image = label(cleared)
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    coords = []
    if len(areas) > 2:
        for region in regionprops(label_image):
            # centroid = region.centroid
            if region.area < 150:
                overlay[region.coords[:,0],region.coords[:,1]] = 1
                ix_inside[region.coords[:,0],region.coords[:,1]] = 0
                #   ax.text(int(centroid[1]), int(centroid[0]), '%4d'%region.area, {'color': 'k', 'fontsize': 10, 'ha': 'center', 'va': 'center',
                # 'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})



    ix_inside = np.stack([ix_inside,ix_inside,ix_inside], axis = 2)

    alpha = np.where(ix_inside,[0.5, 0.5,0.5],[0.0,0.0,0.0])

    ov_inside = np.where(ix_inside, color_inside * alpha, [0.0,0.0,0.0])
    channels =  ov_inside

    idx = alpha  == 0.0

    fig, ax = plt.subplots()




    src = np.where(idx,src ,  channels + (1.0 - alpha) * src )


    contours = measure.find_contours(overlay, bound,positive_orientation = 'high')

    # Display the image and plot all contours found
    # h, w = np.shape(src)[:2]
    # my_dpi = 100



    ax.imshow(np.uint8(src[:,:,::-1]))

    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], color='lime', lw = 2, path_effects=[pe.Stroke(linewidth=4, foreground='navy'),pe.Normal()])#linewidth=2, color='lime')


    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(lab, bbox_inches='tight')
    # im = fig2img(fig)
    # im.save(lab)



def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf



def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return PIL.Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )




def region(image, label):
    ### TODO ###
	# label image regions
	label_image = label(label[:,:,0])
	image_label_overlay = label2rgb(label_image, image=image)

	fig, ax = plt.subplots(figsize=(10, 6))
	ax.imshow(image_label_overlay)

	for region in regionprops(label_image):
		# take regions with large enough areas
		if region.area >= 100:
			# draw rectangle around segmented coins
			minr, minc, maxr, maxc = region.bbox
			rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
									  fill=False, edgecolor='red', linewidth=2)


	ax.set_axis_off()
	plt.tight_layout()
	plt.show()


def random_crop(size):
    # random crop
    np.random.seed(16)
    x =  np.random.randint(low = 0, high = 480 - size - 1)
    y =  np.random.randint(low = 0, high = 640 - size - 1)

    return x, y
