
import vtk
import vtk.util.numpy_support as nps




class MyInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):

    def __init__(self,parent=None):
        self.parent = vtk.vtkRenderWindowInteractor()
        if(parent is not None):
            self.parent = parent

        self.AddObserver("KeyPressEvent",self.keyPressEvent)

    def keyPressEvent(self,obj,event):
        key = self.parent.GetKeySym()
        if key == 'l':
            print(key)
        return




def GetOrientation(caller, ev):
    """
    Print out the orientation.

    We must do this before we register the callback in the calling function.
        GetOrientation.cam = ren.GetActiveCamera()

    :param caller:
    :param ev: The event.
    :return:
    """
    # Just do this to demonstrate who called callback and the event that triggered it.
    print(caller.GetClassName(), "Event Id:", ev)
    # Now print the camera orientation.
    CameraOrientation(GetOrientation.cam)


class OrientationObserver(object):
    def __init__(self, cam):
        self.cam = cam

    def __call__(self, caller, ev):
        # Just do this to demonstrate who called callback and the event that triggered it.
        print(caller.GetClassName(), "Event Id:", ev)
        # Now print the camera orientation.
        CameraOrientation(self.cam)


def CameraOrientation(cam):
    fmt1 = "{:>15s}"
    fmt2 = "{:9.6g}"
    print(fmt1.format("Position:"), ', '.join(map(fmt2.format, cam.GetPosition())))
    print(fmt1.format("Focal point:"), ', '.join(map(fmt2.format, cam.GetFocalPoint())))
    print(fmt1.format("Clipping range:"), ', '.join(map(fmt2.format, cam.GetClippingRange())))
    print(fmt1.format("View up:"), ', '.join(map(fmt2.format, cam.GetViewUp())))
    print(fmt1.format("Distance:"), fmt2.format(cam.GetDistance()))



class Visualizer_3D:
    def __init__(self):
        # self.contour = vtk.vtkContourFilter()
        self.contour = vtk.vtkMarchingCubes()
        self.lut = vtk.vtkLookupTable()
        self.lut.SetTableRange(-1.0, 1.0)
        self.lut.Build()
        #TODO: how to dynamically change this value?
        # it does not update the color once the actor is initalized
        self.contour_actor = vtk.vtkActor()

        self.img_path = "C:\\Users\\ajamgard.1\\Desktop\\TemporalPose\\TestScreenshot.png"
        self.render_window = vtk.vtkRenderWindow()
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

        # self.contour_actor = vtk.vtkLODActor()
        # self.contour_actor.SetNumberOfCloudPoints( 1000 )
        # self.contour_actor.SetMapper( mapper )
        # self.contour_actor.GetProperty().SetColor( 1, 1, 1 )

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

        # render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(_renderer_1)
        self.render_window.AddRenderer(_renderer_2)
        _renderer_1.ResetCamera()
        _renderer_2.ResetCamera()

        _renderer_1.SetViewport(xmins[0], ymins[0], xmaxs[0], ymaxs[0])
        _renderer_2.SetViewport(xmins[1], ymins[1], xmaxs[1], ymaxs[1])

        self.render_window.SetSize(width, height)


        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(self.render_window)

        interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        iren.SetInteractorStyle(interactor_style)

        _renderer_2.SetActiveCamera(_renderer_1.GetActiveCamera())
        _renderer_1.SetBackground( 1, 1,1 )
        _renderer_2.SetBackground( 1, 1,1 )
        # Add a x-y-z coordinate to the original point
        axes_coor = vtk.vtkAxes()
        axes_coor.SetOrigin(0, 0, 0)


        mapper_axes_coor = vtk.vtkPolyDataMapper()
        mapper_axes_coor.SetInputConnection(axes_coor.GetOutputPort())
        actor_axes_coor = vtk.vtkActor()
        actor_axes_coor.SetMapper(mapper_axes_coor)
        _renderer_1.AddActor(actor_axes_coor)
        _renderer_2.AddActor(actor_axes_coor)

        # scale t axis
        # Set up a nice camera position.
        # camera = vtk.vtkCamera()
        # camera.SetPosition( -21.68,   -19.384,  -4.28957)
        # camera.SetFocalPoint(-1.75385,  -17.8939,  -11.0808)
        # camera.SetClippingRange( 1.76526,   47.4309)
        # camera.SetViewUp(0.0848326, -0.995932, 0.0303716)
        # camera.SetDistance( 21.1043)
        # _renderer_2.SetActiveCamera(camera)
        cam = _renderer_2.GetActiveCamera()
        transform = vtk.vtkTransform()
        transform.Scale(10, 1, 1)
        cam.SetModelTransformMatrix(transform.GetMatrix())




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


        # # FIND CAMERA Position
        # GetOrientation.cam = _renderer_1.GetActiveCamera()
        # # Register the callback with the object that is observing.
        # iren.AddObserver('EndInteractionEvent', GetOrientation)

        self.render_window.Render()



        iren.AddObserver('KeyPressEvent', self.keypress_callback, 1.0)

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
        vtk_image_data.SetSpacing([0.01] * 3)  # How to set a correct spacing value??
        vtk_image_data.GetPointData().SetScalars(vtk_data_array)
        vtk_image_data.SetOrigin(-1, -1, -1)

        dims = vtk_image_data.GetDimensions()
        bounds = vtk_image_data.GetBounds()


        implicit_volume = vtk.vtkImplicitVolume()
        implicit_volume.SetVolume(vtk_image_data)

        sample = vtk.vtkSampleFunction()
        sample.SetImplicitFunction(implicit_volume)
        sample.SetSampleDimensions(1000,512,512)
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
        # # We begin by creating the data we want to render.
        # # For this tutorial, we create a 3D-image containing three overlaping cubes.
        # # This data can of course easily be replaced by data from a medical CT-scan or anything else three dimensional.
        # # The only limit is that the data must be reduced to unsigned 8 bit or 16 bit integers.
        #
        #
        # # For VTK to be able to use the data, it must be stored as a VTK-image. This can be done by the vtkImageImport-class which
        # # imports raw data and stores it.
        # dataImporter = vtk.vtkImageImport()
        # # The preaviusly created array is converted to a string of chars and imported.
        # data_string = data_matrix.tostring()
        # dataImporter.CopyImportVoidPointer(data_string, len(data_string))
        # # The type of the newly imported data is set to unsigned char (uint8)
        # dataImporter.SetDataScalarTypeToUnsignedChar()
        # # Because the data that is imported only contains an intensity value (it isnt RGB-coded or someting similar), the importer
        # # must be told this is the case.
        # dataImporter.SetNumberOfScalarComponents(1)
        # # The following two functions describe how the data is stored and the dimensions of the array it is stored in. For this
        # # simple case, all axes are of length 75 and begins with the first element. For other data, this is probably not the case.
        # # I have to admit however, that I honestly dont know the difference between SetDataExtent() and SetWholeExtent() although
        # # VTK complains if not both are used.
        # dataImporter.SetDataExtent(0, 255, 0, 255, 0, 255)
        # dataImporter.SetWholeExtent(0, 255, 0, 255, 0, 255)
        #         # Extract the region of interest
        # voiHead = vtk.vtkExtractVOI()
        # voiHead.SetInputConnection( dataImporter.GetOutputPort() )
        # voiHead.SetVOI(0, 255, 0, 255, 0, 255)
        # voiHead.SetSampleRate( 1,1,1 )
        #         # Generate an isosurface
        # # UNCOMMENT THE FOLLOWING LINE FOR CONTOUR FILTER
        # # contourBoneHead = vtk.vtkContourFilter()
        # # contourBoneHead = vtk.vtkMarchingCubes()
        # # contourBoneHead.SetInputConnection( voiHead.GetOutputPort() )
        # # contourBoneHead.ComputeNormalsOn()
        # # contourBoneHead.SetValue( 0, 50 )  # Bone isovalue
        # #
        # # # Take the isosurface data and create geometry
        # # geoBoneMapper = vtk.vtkPolyDataMapper()
        # # geoBoneMapper.SetInputConnection( contourBoneHead.GetOutputPort() )
        # # geoBoneMapper.ScalarVisibilityOff()
        # #
        # # # Take the isosurface data and create geometry
        # # actorBone = vtk.vtkLODActor()
        # # actorBone.SetNumberOfCloudPoints( 1000000 )
        # # actorBone.SetMapper( geoBoneMapper )
        # # actorBone.GetProperty().SetColor( 1, 1, 1 )
        # #
        # # # Create renderer
        # # ren = vtk.vtkRenderer()
        # # ren.SetBackground( 0.329412, 0.34902, 0.427451 ) #Paraview blue
        # # ren.AddActor(actorBone)
        #
        #
        # # contour = vtk.vtkContourFilter()
        # self.contour.SetInputConnection(voiHead.GetOutputPort())
        # self.contour.SetValue(0, 0.1)
        #
        # dataMapper = vtk.vtkDataSetMapper()
        # dataMapper.SetInputConnection(dataImporter.GetOutputPort())
        # dataActor = vtk.vtkActor()
        # dataActor.SetMapper(dataMapper)
        #
        # # Rendering
        # renderer_1 = vtk.vtkRenderer()  # for tube
        # renderer_2 = vtk.vtkRenderer()  # for contour
        #
        # actor = self.get_actor(self.contour)
        #
        # renderer_1.AddActor(dataActor)
        # renderer_2.AddActor(actor)
        #
        #
        #
        # self.vtk_show(renderer_1,renderer_2)
        #
        # # # Create a window for the renderer of size 250x250
        # # renWin = vtk.vtkRenderWindow()
        # # renWin.AddRenderer(ren)
        # # renWin.SetSize(250, 250)
        # #
        # # # Set an user interface interactor for the render window
        # # iren = vtk.vtkRenderWindowInteractor()
        # # iren.SetRenderWindow(renWin)
        # #
        # # # Start the initialization and rendering
        # # iren.Initialize()
        # # renWin.Render()
        # # iren.Start()
    def keypress_callback(self, obj, ev):
            key = obj.GetKeySym()
            self.CaptureImage()

    def CaptureImage(self):

            w2if = vtk.vtkWindowToImageFilter()
            w2if.SetInput(self.render_window)
            w2if.SetScale(10)
            w2if.SetInputBufferTypeToRGB()
            w2if.ReadFrontBufferOff()
            w2if.Update()

            writer = vtk.vtkPNGWriter()
            writer.SetFileName(self.img_path)
            writer.SetInputConnection(w2if.GetOutputPort())
            writer.Write()
