


import numpy as np
import cv2
from scipy import ndimage
import os


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
        img_output_3d = np.where(inside,np.maximum(-temp,-10), np.minimum(img_output_3d,10))
        img_output_3d = (img_output_3d - (np.min(img_output_3d))) / (np.max(img_output_3d) - np.min(img_output_3d)+ eps)
        # np.savetxt('C:\\Users\\ajamgard.1\\Desktop\\TemporalPose\\tx.txt',img_output_3d[0,:,:], delimiter=',')
        img_output_3d = img_output_3d * 2.0 - 1.0
    return img_output_3d


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


        return img_output_3d, img_input_3d



time_length = 16
base_dir = os.getcwd()
train_lab = prepare_data_refresh(os.path.join(base_dir , "Data\\ReFresh"), time_length)
stacks_train = Stacker(train_lab, time_length)
img_output_1, img_input = stacks_train.get_refresh(250,1)
img_output_3, _ = stacks_train.get_refresh(250, 3)
img_output_10, _ = stacks_train.get_refresh(250, 10)
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("C:\\Users\\ajamgard.1\\Box\\Publications\\Materials\\timeDilation\\output.mp4",fourcc, 5.0, (512*4,512))
for i in range(0,time_length):
    a = img_output_1[i,:,:]
    a = (a - 1.0) / -2.0 * 255.0
    im_color_1 = cv2.applyColorMap(np.uint8(a), cv2.COLORMAP_JET)
    b = img_output_3[i,:,:]
    b = (b - 1.0) / -2.0 * 255.0
    im_color_3 = cv2.applyColorMap(np.uint8(b), cv2.COLORMAP_JET)
    c = img_output_10[i,:,:]
    c = (c - 1.0) / -2.0 * 255.0
    im_color_10 = cv2.applyColorMap(np.uint8(c), cv2.COLORMAP_JET)
    numpy_horizontal = np.hstack((im_color_1,im_color_3, im_color_10, img_input[i,:,:,:]))
    # cv2.imshow('s',numpy_horizontal )
    cv2.imwrite("C:\\Users\\ajamgard.1\\Box\\Publications\\Materials\\timeDilation\\Img-%04d.png"%i,img_input[i,:,:,:] )
    cv2.imwrite("C:\\Users\\ajamgard.1\\Box\\Publications\\Materials\\timeDilation\\T1-%04d.png"%i,im_color_1 )
    cv2.imwrite("C:\\Users\\ajamgard.1\\Box\\Publications\\Materials\\timeDilation\\T3-%04d.png"%i,im_color_3 )
    cv2.imwrite("C:\\Users\\ajamgard.1\\Box\\Publications\\Materials\\timeDilation\\T10-%04d.png"%i,im_color_10 )

    # cv2.waitKey(0)
    # out.write(numpy_horizontal)
# out.release()
cv2.destroyAllWindows()
