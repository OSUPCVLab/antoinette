
"""
Transparent Image overlay(Alpha blending) with OpenCV and Python

"""



import cv2

import numpy as np

from scipy import ndimage


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


# function to overlay a transparent image on background.

def transparentOverlay(src , overlay , pos=(0,0),scale = 1):

	"""

	:param src: Input Color Background Image

	:param overlay: transparent Image (BGRA)

	:param pos:  position where the image to be blit.

	:param scale : scale factor of transparent image.

	:return: Resultant Image

	"""

	# overlay = cv2.resize(overlay,(0,0),fx=scale,fy=scale)

	h,w= overlay.shape  # Size of foreground

	rows,cols,_ = src.shape  # Size of background Image

	x, y = pos[0],pos[1]    # Position of foreground/overlay image


	# alpha = np.zeros((h,w,3), dtype = np.float32)
	# r_start = x
	# r_end = min(x+h,rows)
	#
	# c_start = y
	# c_end = min(y+w,cols)

	#
	# src[r_start:r_end,c_start:c_end] = \
	# 								channels[:r_end-r_start,:c_end-c_start] +\
	# 								(1.0 - alpha[:r_end-r_start,:c_end-c_start]) *\
	# 								src[r_start:r_end,c_start:c_end]


	# alpha[:,:,0] = overlay[:,:,3]/255.0
	# alpha[:,:,1] = overlay[:,:,3]/255.0
	# alpha[:,:,2] = overlay[:,:,3]/255.0
	#
	# channels = alpha * overlay[:,:,:3]

	# idx = alpha  == 0.0
	# src = np.where(idx,src ,  channels + (1.0 - alpha) * src )

	color_contour = [0,255,0]
	color_inside = [255,255,255]

	ix_contour = np.logical_and(overlay < 0.2 , overlay > -0.2)
	ix_inside = overlay > 	0.2

	ix_contour = np.stack([ix_contour,ix_contour,ix_contour], axis = 2)
	ix_inside = np.stack([ix_inside,ix_inside,ix_inside], axis = 2)

	alpha_contour = np.where(ix_contour, [1.0,1.0,1.0], [0.0,0.0,0.0])
	alpha_inside = np.where(ix_inside,[0.7, 0.7,0.7],[0.0,0.0,0.0])


	alpha = alpha_contour + alpha_inside


	ov_contour = np.where(ix_contour, color_contour * alpha, [0.0,0.0,0.0])
	ov_inside = np.where(ix_inside, color_inside * alpha, [0.0,0.0,0.0])
	channels = ov_contour + ov_inside

	idx = alpha  == 0.0
	src = np.where(idx,src ,  channels + (1.0 - alpha) * src )

	return np.uint8(src)



# read all images

bImg = cv2.imread("foreground.jpg")


# KeyPoint : Remember to use cv2.IMREAD_UNCHANGED flag to load the image with alpha channel
logoImage = np.zeros((bImg.shape[0],bImg.shape[1]), dtype = np.uint8)
# logoImage[10:150, 10:150] = 1#[0,0,0, 100]
# logoImage[10:150, 10:12, :] = [0,255,255, 255]
# logoImage[10:12, 10:150, :] = [0,255,255, 255]
# logoImage[148:150, 10:150, :] = [0,255,255, 255]
# logoImage[0:150, 148:150, :] = [0,255,255, 255]


x = np.arange(0, bImg.shape[0])
y = np.arange(0, bImg.shape[1])


for i in range(len(x)):
	for j in range(len(y)):
			if (i - len(x)/2)**2 + (j - len(y)/2)**2  - 1000 <= 0:
				logoImage[i,j] = 255


out = distance_transform(logoImage, mode = 'thresh-signed')

result = transparentOverlay(bImg,out,(0,0), 1 )



#Display the result

cv2.namedWindow("Result")

cv2.imshow("Result",result)

cv2.waitKey()

cv2.destroyAllWindows()
