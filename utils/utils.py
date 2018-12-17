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
from pycoco.coco import COCO
from itertools import islice
import datetime
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
from scipy import ndimage
import matplotlib.pyplot as plt
import vtk
import vtk.util.numpy_support as nps
from pathlib import Path, PureWindowsPath

def prepare_data(dataset_dir):
    train_input_names=[]
    train_output_names=[]
    val_input_names=[]
    val_output_names=[]
    test_input_names=[]
    test_output_names=[]
    for file in os.listdir(dataset_dir + "\\images\\train"):
        cwd = os.getcwd()
        train_input_names.append(cwd + "\\" + dataset_dir + "\\images\\train\\" + file)

    for file in os.listdir(dataset_dir + "\\annotations\\train"):
        cwd = os.getcwd()
        train_output_names.append(dataset_dir + "\\annotations\\train\\" + file)

    for file in os.listdir(dataset_dir + "\\images\\val"):
        cwd = os.getcwd()
        val_input_names.append(cwd + "\\" + dataset_dir + "\\images\\val\\" + file)

    for file in os.listdir(dataset_dir + "\\annotations\\val"):
        cwd = os.getcwd()
        val_output_names.append( dataset_dir + "\\annotations\\val\\" + file)

    for file in os.listdir(dataset_dir + "\\images\\test"):
        cwd = os.getcwd()
        test_input_names.append(cwd + "\\" + dataset_dir + "\\images\\test\\" + file)

    for file in os.listdir(dataset_dir + "\\annotations\\test"):
        cwd = os.getcwd()
        test_output_names.append(dataset_dir + "\\annotations\\test\\" + file)

    train_input_names.sort(),train_output_names.sort(), val_input_names.sort(), val_output_names.sort(), test_input_names.sort(), test_output_names.sort()
    #TODO: Find a better way to remove .D_Store folder
    return train_output_names[1:],\
           val_output_names[1:], \
           test_output_names[1:]\


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
class DataGenerator:
    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}


    #  From COCO
    def add_image( self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    #  From COCO
    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def load_annotation(self, seq_name):
        coco = COCO(seq_name)

        class_ids = sorted(coco.getCatIds())

        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id]))) #catIds=[id]->goes inside coco.getImgIds
            # Remove duplicates
            # image_ids = list(set(image_ids))
        # image_ids = list(coco.imgs.keys())

        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])
        # Add images
        image_dir = "C:\\Users\\ajamgard.1\\Desktop\\TemporalPose\\Data\\PoseTrack"
        # image_dir = "E:\\Datasets\\Data\\PoseTrack"
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir ,PureWindowsPath(coco.imgs[i]['file_name'])),
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))



    def load_img(self):
        """ Forget this function for now """
        # read through the image folder
        colors = self.get_spaced_colors(17)

        images = []
        img = cv2.imread(self.image_info[0]['path'])
        xt_slice  = np.zeros((len(self.image_info),img.shape[0], 3), dtype = np.uint8)
        yt_slice  = np.zeros((len(self.image_info),img.shape[1],3) , dtype = np.uint8)
        print(np.shape(xt_slice))
        for i in range(len(self.image_info)):
            img = cv2.imread(self.image_info[i]['path'])

            xt_slice[i,:,:] = img[:,360,:].reshape((1,img.shape[0],3))
            yt_slice[i,:,:] = img[360,:,:].reshape((1,img.shape[1],3))


            for j in range(len(self.image_info[i]['annotations'])):
                keypoints = self.image_info[i]['annotations'][j]['keypoints']
                for k in range(2,51,3):
                    if keypoints[k] == 1:
                        xt_slice[i,int(keypoints[k-1]),:] = colors[k//3]
                        yt_slice[i,int(keypoints[k-2]),:] = colors[k//3]


        return yt_slice, xt_slice
class Visualizer_3D:
    def __init__(self):
        self.contour = vtk.vtkContourFilter()
        self.lut = vtk.vtkLookupTable()
        self.lut.SetTableRange(0.0, 1.0)
        self.lut.Build()
        #TODO: how to dynamically change this value?
        # it does not update the color once the actor is initalized
        self.contour_actor = vtk.vtkActor()
    def get_actor(self,vtk_source):

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(vtk_source.GetOutputPort())
        normals.SetFeatureAngle(60.0)
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
        min = 0.0
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
        num_array=data_matrix.transpose(2, 0, 1).ravel(),  # ndarray contains the fitting result from the points. It is a 3D array
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




class Stacker:
    def __init__(self, info, time_length):
        self.info = info
        self.time_length =time_length

    def generate_distance_transform(self, image_info, frame_number):
        """ Genereate Distance Transform from joints
            by creating thick skeletons
        """
        # How limbs should be connected -> Look at .json file
        # in categories:skeleton
        limbs_links = np.array([[2,0], [0,1],[5,7], [6,8], [7,9], [8,10],
                          [11,13],[12,14], [13,15], [14,16]])

        # How torso should be constructed
        torso_links = np.array([1,5,11,12,6,1])



        img = cv2.imread(image_info[frame_number]['path'])

        img_final = np.ones((img.shape[0],img.shape[1]), dtype = np.uint8)
        for j in range(len(image_info[frame_number]['annotations'])):
            img_temp = np.ones((img.shape[0],img.shape[1]), dtype = np.uint8)
            keypoints = image_info[frame_number]['annotations'][j]['keypoints']
            for l in limbs_links:
                if keypoints[l[0]*3+2]*keypoints[l[1]*3+2] == 1:
                    cv2.line(img_temp, (int(keypoints[l[0]*3]),int(keypoints[l[0]*3+1])),
                                  (int(keypoints[l[1]*3]),int(keypoints[l[1]*3+1])),
                                  (0,0,0),10)
            path = []

            start_i = 0

            while start_i < len(torso_links)-1:
                if keypoints[torso_links[start_i]*3+2] != 1:
                    start_i += 1
                    continue

                end_i = start_i + 1
                while end_i < len(torso_links)-1:
                    if keypoints[torso_links[end_i]*3+2] != 1:
                        end_i += 1
                        continue
                    break
                start = torso_links[start_i]
                end = torso_links[end_i]
                path.append([keypoints[start*3],keypoints[start*3+1]])
                if keypoints[end*3+ 2] != 1:
                    path.append(path[0])
                else:
                    path.append([keypoints[end*3],keypoints[end*3+1]])
                start_i = end_i



            path = np.array(path, dtype = np.int32)
            path = path.reshape((-1,1,2))

            cv2.fillConvexPoly(img_temp, path,  (0,0,0))

            # 1) background = 0, foreground = 1 -> a) sum(DT(img_i))
            # img_final += cv2.distanceTransform(img_temp, cv2.DIST_L2, 3)

            # 1) background = 0, foreground = 1 ->b) DT(sum(img_i))
            # img_final += img_temp

            # 2) background = 1, 3 = 0 -> a) DT(prod(img_i)) -> img_final = ones
            img_final *= img_temp
            # 2) background = 1, foreground = 0 -> b) sum(DT(img_i))
            # img_final += cv2.distanceTransform(img_temp, cv2.DIST_L2, 3)

        #1.b) & 2.a) continue
        img_final = cv2.resize(img_final,(128,128))#, fx=0.5, fy=0.5)
        img_final = np.array(img_final, dtype = np.uint8)
        # img_final  = np.array(cv2.distanceTransform(img_final, cv2.DIST_L1, 3), dtype = np.uint8)
        img = cv2.resize(img,(128,128))#, fx=0.5, fy=0.5)

        # Uncomment only for visualization purposes
        # img_final = cv2.normalize(img_final,  0, 255, cv2.NORM_MINMAX)

        return img_final, img

    def vectorize_stack_images(self, seq):
        """ Vectorize and Stack a sequence of images """
        #TODO: change this from config file
        img_input = []
        for i in range(self.time_length):
            img_o, img_i = self.generate_distance_transform(self.info[seq[0]],
                                                            int(seq[1]) + i)

            img_input.append(img_i)
        return img_o, img_input


    def vectorize_stack_images_lstm(self, seq):
        """ Vectorize and Stack a sequence of images """
        #TODO: change this from config file
        img_output = np.empty((self.time_length, 128*128), dtype = np.float32)
        img_input = np.empty((self.time_length, 128*128,3), dtype = np.uint8)
        for i in range(self.time_length):
            img_o, img_i = self.generate_distance_transform(self.info[seq[0]],
                                                            int(seq[1]) + i)
            img_o = np.reshape(img_o, (1,img_o.shape[0]*img_o.shape[1]))
            img_i = np.reshape(img_i, (1,img_i.shape[0]*img_i.shape[1],3))
            img_output[i] = img_o
            img_input[i] = img_i

    def distance_transform_3d(self, img_input):

        # # Prepare the embedding function.
        dist_func = ndimage.distance_transform_edt
        # f = dist_func(img_input) + dist_func(1 - img_input)

        # print(np.max(f))
        # print(np.min(f))
        # f= f == 1
        # # Signed distance transform
        # distance = np.where(f, 0, dist_func(img_input) + dist_func(1 - img_input))
        # print(np.max(distance))
        # print(np.min(distance))
        return dist_func(img_input)




    def vectorize_stack_images_2d_temporal(self, seq):
        """ Vectorize and Stack a sequence of images """
        #TODO: change this from config file
        img_output_h = np.zeros((32, 4096), dtype = np.uint8)# cutting the w axis
        img_input_h = np.zeros((32, 4096,3), dtype = np.uint8)# cutting the w axis
        img_output_w = np.zeros((32, 4096), dtype = np.uint8)# cutting the h axis
        img_input_w = np.zeros((32, 4096,3), dtype = np.uint8)
        img_output_3d = np.zeros((64, 64,20), dtype = np.uint8)




        # print(len(self.info[seq[0]]))
        for i in range(self.time_length):
            img_o, img_i = self.generate_distance_transform(self.info[seq[0]],
                                                            int(seq[1]) + i)
            # img_o can be calculated via 2d or 3d distance transform
            #TODO: unify where distance transform is calcualted
            img_output_3d[:,:,i] = img_o
            # flip every other row to create a continuous reshape
            img_i_h = image_serialization([64,64,3],img_i)
            img_i_h = np.reshape(img_i_h, (1,img_i.shape[0]*img_i.shape[1],3))
            # img_o = np.reshape(img_o, (1,img_o.shape[0]*img_o.shape[1]))
            # img_output[i] = img_o
            img_input_h[i] = img_i_h

            img_i_w = image_serialization([64,64,3],np.transpose(img_i,(1,0,2)))
            img_i_w = np.reshape(img_i_w, (1,img_i.shape[0]*img_i.shape[1],3))
            # img_o = np.reshape(img_o, (1,img_o.shape[0]*img_o.shape[1]))
            # img_output[i] = img_o
            img_input_w[i] = img_i_w

        img_output_3d = self.distance_transform_3d(img_output_3d)
        for i in range(self.time_length):
            img_o = img_output_3d[:,:,i]
            img_o_h =  image_serialization([64,64,1],img_o)
            img_o_w =  image_serialization([64,64,1],np.transpose(img_o))
            img_o_h = np.reshape(img_o_h, (1,img_o.shape[0]*img_o.shape[1]))
            img_o_w = np.reshape(img_o_w, (1,img_o.shape[0]*img_o.shape[1]))
            img_output_h[i] = img_o_h
            img_output_w[i] = img_o_w
        img_output_h = (img_output_h - (np.min(img_output_h))) / (np.max(img_output_h) - np.min(img_output_h))
        # img_output_W = (img_output_W - (np.min(img_output_W))) / (np.max(img_output_W) - np.min(img_output_W))
        # for i in range(20):
            # img_input_3d[:,:,i] = cv2.normalize(img_input_3d[:,:,i],  0, 255, cv2.NORM_MINMAX)
        # cv2.imshow('s',img_output_3d[:,:,0])
        # Visualizer_3D().visualize_3d_volume(img_output_3d)
            # cv2.waitKey(1)
        # cv2.waitKey(0)
        # print(np.max(img_output_3d))
        # print(np.min(img_output_3d))
        # cv2.imshow('input_w', img_output_w*255.0)
        # cv2.imshow('input_h', img_output_h*255.0)
        # cv2.imshow('mask', cv2.normalize(img_input_3d[:,20,:],  0, 255, cv2.NORM_MINMAX))
        # cv2.waitKey(0)
        #TODO: think about how to use both outputs
        return img_output_h, [img_input_h,img_input_w]
        # return img_output_h, img_input_h




    def vectorize_stack_images_3d_temporal(self, seq):

        """ Vectorize and Stack a sequence of images """
        img_input_3d = np.zeros((self.time_length,128, 128,3), dtype = np.uint8)
        img_output_3d = np.zeros((self.time_length,128, 128), dtype = np.uint8)


        for i in range(self.time_length):
            img_o, img_i = self.generate_distance_transform(self.info[seq[0]],int(seq[1]) + i)
            img_input_3d[i,:,:,:] = img_i
            img_output_3d[i,:,:] = img_o
        img_output_3d = self.distance_transform_3d(img_output_3d)
        img_output_3d = (img_output_3d - (np.min(img_output_3d))) / (np.max(img_output_3d) - np.min(img_output_3d))

        return img_output_3d, img_input_3d


def generate_labels_with_permutation(dataset, time_length):
    file_names = []
    annotations = {}
    # i = 0
    for seq in dataset:
        print(seq)
        ann = DataGenerator()
        ann.load_annotation(seq)
        annotations[seq] = ann.image_info
        # for i in range(len(ann.image_info)-time_length):
        file_names.append([seq, 0])
        # if i == 40:
           # break
        # i+=1
    return file_names, annotations

# def temporal_data_generator(dataset, time_length = 3):
#     """" train_input_names,train_output_names,
#         val_input_names, val_output_names,
#         test_input_names, test_output_names
#     """
#     # Read all the file names from the provided dataset
#     # train_ann, val_ann, test_ann = prepare_data(dataset_dir=dataset)
#
#     file_map = {}
#     var_count = 0
#     DataGenerator().vectorize_stack_images(dataset,)
#
#     temporal_filename = []
#
#         # For each unique sequence generate a 'time_length' size temporal image
#
#     return file_map[0], file_map[1], file_map[2], file_map[3],file_map[4],file_map[5]





        # self.image_info = []




# Print with time. To console or file
def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)

def inverse_distance_transform(img_gt, img_pred):
    return img_gt + img_pred, img_gt - img_pred



# Compute the average segmentation accuracy across all classes
def compute_global_accuracy(pred, label):
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

# Compute the class-specific segmentation accuracy
def compute_class_accuracies(pred, label, num_classes):
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT,
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies


def compute_mean_iou(pred, label):

    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels);

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))


    mean_iou = np.mean(I / U)
    return mean_iou


def evaluate_segmentation(pred, label, num_classes, score_averaging="weighted"):
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)

    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging)
    f1 = f1_score(flat_pred, flat_label, average=score_averaging)

    iou = compute_mean_iou(flat_pred, flat_label)

    return global_accuracy, class_accuracies, prec, rec, f1, iou
# Takes an absolute file path and returns the name of the file without th extension
def filepath_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name
def get_spaced_colors(n):
    """ Makes a pallete """
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

def image_serialization(shape, img):
    #TODO: remove redundancy in calcualting f again and again
    # Image serilization
    f = (np.arange(0,shape[0]) % 2 == 1).transpose()
    if shape[2] == 1:
        f = np.tile(f[:, None], (1,shape[1]))
    else:
        f = np.tile(f[:, None, None], (1,shape[1],shape[2]))
    return np.where(f, img, np.fliplr(img))
