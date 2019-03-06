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
import vtk
import vtk.util.numpy_support as nps
import random
import ntpath
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
    for path, subdirs, files in os.walk(dataset_dir + "\\val"):
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
    train_ind = random.sample(range(0,len(train_names['input'])),2000)
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
#
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
            if ii >= frame_number*(self.time_length-1):
                img_input_3d[i,:,:,:] = cv2.resize(frame, (size,size))
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

        h = [0.5,0.,-0.5]
        w = [0.5,0.,-0.5]
        t = [0.5,0.,-0.5]
        tt, hh, ww = np.meshgrid(t,h,w,indexing='ij')
        dt = ndimage.convolve(img_output_3d, tt)
        dh = ndimage.convolve(img_output_3d, hh)
        dw = ndimage.convolve(img_output_3d, ww)
        gradients = np.stack([dt,dh,dw],3)
        gradients[:,:,:,0] /= (np.linalg.norm(gradients,axis = 3) + 1e-15)
        gradients[:,:,:,1] /= (np.linalg.norm(gradients,axis = 3) + 1e-15)
        gradients[:,:,:,2] /= (np.linalg.norm(gradients,axis = 3) + 1e-15)
        # for i in range(16):
            # img_output_3d[i,:,:,0] = cv2.normalize(img_output_3d[i,:,:,0],  0, 255, cv2.NORM_MINMAX)
        # cv2.imshow('s',img_output_3d[0,:,:,0])
        # Visualizer_3D().visualize_3d_volume(img_output_3d[:,:,:,1])
        #
        # cv2.waitKey(0)
        # img_output_3d = np.stack((img_output_3d_img,img_output_3d_impl),axis = 3)
        return img_output_3d, img_input_3d, gradients

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

class Visualizer_3D:
    def __init__(self):
        self.contour = vtk.vtkContourFilter()
        self.lut = vtk.vtkLookupTable()
        self.lut.SetTableRange(-1.0, 1.0)
        self.lut.Build()
        #TODO: how to dynamically change this value?
        # it does not update the color once the actor is initalized
        self.contour_actor = vtk.vtkActor()
    def get_actor(self,vtk_source):

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(vtk_source.GetOutputPort())
        normals.SetFeatureAngle(30.0)
        normals.ReleaseDataFlagOn()

        stripper = vtk.vtkStripper()
        stripper.SetInputConnection(normals.GetOutputPort())
        stripper.ReleaseDataFlagOn()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(stripper.GetOutputPort())
        mapper.SetScalarVisibility(0)


        self.contour_actor = vtk.vtkActor()
        self.contour_actor.SetMapper(mapper)

        self.contour_actor.GetProperty().SetDiffuseColor([1,1,0])
        self.contour_actor.GetProperty().SetSpecular(0.3)
        self.contour_actor.GetProperty().SetSpecularPower(20)

        return self.contour_actor

    def vtkSliderCallback2(self,obj, event):
        sliderRepres = obj.GetRepresentation()
        pos = sliderRepres.GetValue()

        self.contour.SetValue(0, pos)
        color =[0]*3
        self.lut.GetColor(pos, color)
        self.contour_actor.GetProperty().SetDiffuseColor(color)

    def vtk_show(self,_renderer_1, _renderer_2, width=640 * 2, height=480):

        # Multiple Viewports
        xmins = [0.0, 0.5]
        xmaxs = [0.5, 1.0]
        ymins = [0.0, 0.0]
        ymaxs = [1.0, 1.0]

        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(_renderer_1)
        render_window.AddRenderer(_renderer_2)
        _renderer_1.ResetCamera()
        _renderer_2.ResetCamera()

        _renderer_1.SetViewport(xmins[0], ymins[0], xmaxs[0], ymaxs[0])
        _renderer_2.SetViewport(xmins[1], ymins[1], xmaxs[1], ymaxs[1])

        render_window.SetSize(width, height)

        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(render_window)

        interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        iren.SetInteractorStyle(interactor_style)

        _renderer_2.SetActiveCamera(_renderer_1.GetActiveCamera())

        # Add a x-y-z coordinate to the original point
        axes_coor = vtk.vtkAxes()
        axes_coor.SetOrigin(0, 0, 0)
        mapper_axes_coor = vtk.vtkPolyDataMapper()
        mapper_axes_coor.SetInputConnection(axes_coor.GetOutputPort())
        actor_axes_coor = vtk.vtkActor()
        actor_axes_coor.SetMapper(mapper_axes_coor)
        _renderer_1.AddActor(actor_axes_coor)
        _renderer_2.AddActor(actor_axes_coor)


        # create the scalar_bar
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetOrientationToHorizontal()
        scalar_bar.SetLookupTable(self.lut)

        # create the scalar_bar_widget
        scalar_bar_widget = vtk.vtkScalarBarWidget()
        scalar_bar_widget.SetCurrentRenderer(_renderer_1)
        scalar_bar_widget.SetInteractor(iren)
        scalar_bar_widget.SetScalarBarActor(scalar_bar)
        scalar_bar_widget.On()


        SliderRepres = vtk.vtkSliderRepresentation2D()
        min = -1.0
        max = 1.0
        SliderRepres.SetMinimumValue(min)
        SliderRepres.SetMaximumValue(max)
        SliderRepres.SetValue((min + max) / 2)
        # SliderRepres.SetTitleText("Slider")
        SliderRepres.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        SliderRepres.GetPoint1Coordinate().SetValue(0.1 , 0.9)
        SliderRepres.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        SliderRepres.GetPoint2Coordinate().SetValue(0.4 , 0.9)
        SliderRepres.SetSliderLength(0.02)
        SliderRepres.SetSliderWidth(0.06)
        SliderRepres.SetEndCapLength(0.04)
        SliderRepres.SetEndCapWidth(0.04)
        SliderRepres.SetTubeWidth(0.01)
        SliderRepres.SetLabelFormat("%6.3g")
        SliderRepres.SetTitleHeight(0.02)
        SliderRepres.SetLabelHeight(0.02)

        SliderWidget = vtk.vtkSliderWidget()
        SliderWidget.SetCurrentRenderer(_renderer_2)
        SliderWidget.SetInteractor(iren)
        SliderWidget.SetRepresentation(SliderRepres)
        SliderWidget.KeyPressActivationOff()
        SliderWidget.SetAnimationModeToAnimate()
        SliderWidget.SetEnabled(True)
        SliderWidget.AddObserver("EndInteractionEvent", self.vtkSliderCallback2)


        iren.Initialize()
        iren.Start()

    def visualize_3d_volume(self, data_matrix):

        # Create the standard renderer, render window
        # and interactor.
        vtk_data_array = nps.numpy_to_vtk(
        num_array=data_matrix.transpose(2, 1, 0).ravel(),  # ndarray contains the fitting result from the points. It is a 3D array
        deep=True,
        array_type=vtk.VTK_FLOAT)

        # Convert vtkFloatArray to vtkImageData
        vtk_image_data = vtk.vtkImageData()
        vtk_image_data.SetDimensions(data_matrix.shape)
        vtk_image_data.SetSpacing([0.1] * 3)  # How to set a correct spacing value??
        vtk_image_data.GetPointData().SetScalars(vtk_data_array)
        vtk_image_data.SetOrigin(-1, -1, -1)

        dims = vtk_image_data.GetDimensions()
        bounds = vtk_image_data.GetBounds()

        implicit_volume = vtk.vtkImplicitVolume()
        implicit_volume.SetVolume(vtk_image_data)

        sample = vtk.vtkSampleFunction()
        sample.SetImplicitFunction(implicit_volume)
        sample.SetModelBounds(bounds)
        sample.ComputeNormalsOff()

        # contour = vtk.vtkContourFilter()
        self.contour.SetInputConnection(sample.GetOutputPort())
        self.contour.SetValue(0, 0.1)

        dataMapper = vtk.vtkDataSetMapper()
        dataMapper.SetInputConnection(sample.GetOutputPort())
        dataActor = vtk.vtkActor()
        dataActor.SetMapper(dataMapper)

        # Rendering
        renderer_1 = vtk.vtkRenderer()  # for tube
        renderer_2 = vtk.vtkRenderer()  # for contour

        actor = self.get_actor(self.contour)

        renderer_1.AddActor(dataActor)
        renderer_2.AddActor(actor)


        self.vtk_show(renderer_1,renderer_2)


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
