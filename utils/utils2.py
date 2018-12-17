"""
Temporal Pose
Common utility functions and classes.

Copyright (c) 2018 PCVLab & ADL
Licensed under the MIT License (see LICENSE for details)
Written by Nima A. Gard
"""

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



class Stacker:
    def __init__(self, info, time_length):
        self.info = info
        self.time_length =time_length
    def distance_transform(self, vol, mode ='unsigned'):

        img_output_3d = ndimage.distance_transform_edt(vol)
        img_output_3d = (img_output_3d - (np.min(img_output_3d))) / (np.max(img_output_3d) - np.min(img_output_3d))
        if mode == 'signed':
            inside = vol == 0.0
            temp = ndimage.distance_transform_edt(1 - vol)
            temp = (temp - (np.min(temp))) / (np.max(temp) - np.min(temp))
            img_output_3d = np.where(inside,-temp, img_output_3d)
        # np.savetxt('C:\\Users\\ajamgard.1\\Desktop\\TemporalPose\\tx.txt',img_output_3d[0,:,:], delimiter=',')
#
        return img_output_3d




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





        img_output_3d =  self.distance_transform(img_output_3d, mode ='signed')
        # for i in range(16):
        #     img_output_3d[i,:,:] = cv2.normalize(img_output_3d[i,:,:],  0, 255, cv2.NORM_MINMAX)
        # cv2.imshow('s',img_output_3d[0,:,:])
        # Visualizer_3D().visualize_3d_volume(img_output_3d)
            # cv2.waitKey(1)
        # cv2.waitKey(0)
        return img_output_3d, img_input_3d

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




# Print with time. To console or file
def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)
