#------
# Nima A. Gard
#------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cv2
import numpy as np
import random
import time, datetime
from utils import utils2 as utils, helpers
from builders import model_builder
import os
# from skimage import color
# use 'Agg' on matplotlib so that plots could be generated even without Xserver
# running
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score, precision_recall_curve
from sklearn.metrics import auc
EPOCHS = 1000




def overlay(img, mask, alpha = 0.6):
    img_hsv = color.rgb2hsv(img)
    color_mask_hsv = color.rgb2hsv(mask)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)


    return img_masked
def mask_color_img(img, mask, color=[0, 255, 255], alpha=0.6):
    '''
    img: cv2 image
    mask: bool or np.where
    color: BGR triplet [_, _, _]. Default: [0, 255, 255] is yellow.
    alpha: float [0, 1].

    Ref: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    '''
    out = img.copy()
    img_layer = img.copy()
    mask_layer = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)  # mask is only
    mask_layer[mask[:,:,0]] = 1
    # mask_layer[mask] = 1
    fg = cv2.bitwise_or(img_layer, np.array(color), mask=mask_layer)
    # img_layer[mask] = color_mask[mask]
    out = cv2.addWeighted(fg, alpha, out, 1 - alpha, 0, out)
    return(out)

def compute_normlas(inputs):

	# dx = tf.layers.conv3d(inputs, 1, (3,1,1), strides =(1,1,1), activation=None , padding = 'same')
	# dy = tf.layers.conv3d(inputs, 1, (1,3,1), strides =(1,1,1), activation=None , padding = 'same')
	# dz = tf.layers.conv3d(inputs, 1, (1,1,3), strides =(1,1,1), activation=None , padding = 'same')
	h = [0.5,0.,-0.5]
	w = [0.5,0.,-0.5]
	t = [0.5,0.,-0.5]
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


	dt = tf.nn.conv3d(inputs, tt, strides=[1,1,1,1,1], padding='SAME', name ='convT')
	dh = tf.nn.conv3d(inputs, hh, strides=[1,1,1,1,1], padding='SAME', name ='convH')
	dw = tf.nn.conv3d(inputs, ww, strides=[1,1,1,1,1], padding='SAME', name ='convW')
	normals = tf.concat([dt, dh, dw],4)

	# normalizing causes loss to go nan!!!!
	#norm = tf.norm(normals, axis = 4, keepdims=True) + 1e-15
	#dt /= norm
	#dh /= norm
	#dw /= norm

	#normals = tf.concat([dt, dh, dw],4)

	return normals


def main():
	base_dir = os.getcwd()

	time_length = 8
	# train_lab, val_lab, test_lab = utils.prepare_data(os.path.join(base_dir , "Data\\SURREAL"), time_length, 'TEST')
	# train_lab, val_lab, test_lab = utils.prepare_data_synthia(os.path.join(base_dir , "Data\\SYNTHIA-SEQS-01-SUMMER"), time_length)
	# test_lab = utils.prepare_video(os.path.join(base_dir , "Data"), time_length, 'TEST')
	train_lab, val_lab, test_lab = utils.prepare_data_refresh(os.path.join(base_dir , "Data\\ReFresh"), time_length)
	# train_lab, val_lab, test_lab = utils.prepare_data_posetrack(os.path.join(base_dir , "E:\\Datasets\\Nima\\PoseTrack\\images"), time_length)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config = config)
	num_classes =  1
	net_input =  tf.placeholder(tf.float32,shape=[None,time_length,None,None,3])
	net_output = tf.placeholder(tf.float32,shape=[None,time_length,None,None,1])#,None,num_classes
	net_normals = compute_normlas(net_output)
	# network, network_normals = model_builder.build_model(model_name='UNet-3D',frontend ='ResNet101', net_input=net_input, num_classes=num_classes)

	network,_ = model_builder.build_model(model_name='UNet-3D',frontend ='ResNet101', net_input=net_input, num_classes=num_classes)
	# loss = tf.reduce_mean(tf.nn.l2_loss(network- net_output))#softmax_cross_entropy_with_logits_v2(logits = network, labels = net_output)




	# opt = tf.train.AdamOptimizer(learning_rate = 0.001 ).minimize(loss, var_list = [var for var in tf.trainable_variables()])

	saver = tf.train.Saver(max_to_keep  = 1000)
	sess.run(tf.global_variables_initializer())

	model_checkpoint_name = os.path.join(base_dir , "checkpoints\\latest_model_EncoderDecoder.ckpt")
	saver.restore(sess, model_checkpoint_name)



	stacks_test = utils.Stacker(test_lab, time_length)
	batch_size  = 1
	label_values = []


	per_seq_acc = []
	per_seq_prec = []
	per_seq_rec = []
	per_seq_f1 = []
	per_seq_iou = []
    #
    #
	score=open("%s\\scores.csv"%("results"),'w')
	score.write("avg_accuracy, precision, recall, f1 score, mean iou\n" )
	for ind in range(0,len(test_lab['input'])):#
		gt, input_image = stacks_test.get_refresh_v3(ind,3)#get_data(ind)#stacks_test.get_refresh(ind)#
		input_image = np.expand_dims(input_image,axis=0)
		# output_image, output_normals = sess.run([network, network_normals], feed_dict = {net_input: input_image/255.0})
		# output_image, output_normals = sess.run([network, compute_normlas(network)], feed_dict = {net_input: input_image/255.0})
		output_image = sess.run(network, feed_dict = {net_input: input_image/255.0})

		# gt_temp = np.float32(np.expand_dims(gt, axis = 3))
#
		# gt_normals = sess.run(net_normals, feed_dict = {net_output: [gt_temp]})

		output_image = np.array(output_image[0,:,:,:])




		if not os.path.isdir(os.path.join(base_dir , "%s\\%04d"%("results",ind))):
			os.makedirs(os.path.join(base_dir , "%s\\%04d"%("results",ind)))

		# SAVE scors
		target=open("%s\\%04d\\val_scores.csv"%("results",ind),'w')
		target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou\n" )




		per_img_acc = []
		per_img_prec = []
		per_img_rec = []
		per_img_f1 = []
		per_img_iou = []


		for i in range(time_length):
			# gt_i = gt[i,:,:]*255.0
			# output_image_i = output_image[i,:,:]*255.0

			gt_i = gt[i,:,:]
			gt_i_color = (gt_i - 1.0) / -2.0
			gt_i_color = cv2.applyColorMap(np.uint8(gt_i_color*255.0), cv2.COLORMAP_JET)

			output_image_i = output_image[i,:,:]
			output_image_color = (output_image_i - 1.0) / -2.0

			output_image_color = cv2.applyColorMap(np.uint8(output_image_color*255.0), cv2.COLORMAP_JET)

			input_image_i = input_image[0,i,:,:,:]


            #
			# output_normals_i = output_normals[0,i,:,:,:]
			# output_normals_l2_norm = np.linalg.norm(output_normals_i,axis = 2) + 1e-12
			# output_normals_i[:,:,0] /= output_normals_l2_norm
			# output_normals_i[:,:,1] /= output_normals_l2_norm
			# output_normals_i[:,:,2] /= output_normals_l2_norm
			# output_normals_i = (output_normals_i + 1.0) * 0.5 * 255.0
            #
			# gt_normals_i = gt_normals[0,i,:,:,:]
			# gt_normals_l2_norm = np.linalg.norm(gt_normals_i,axis = 2) + 1e-12
			# gt_normals_i[:,:,0] /= gt_normals_l2_norm
			# gt_normals_i[:,:,1] /= gt_normals_l2_norm
			# gt_normals_i[:,:,2] /= gt_normals_l2_norm
			# gt_normals_i = (gt_normals_i + 1.0) * 0.5 * 255.0

			file_name = os.path.basename(test_lab['input'][0])#[ind][0])
			file_name = os.path.splitext(file_name)[0]

			cv2.imwrite(os.path.join(base_dir ,"%s\\%04d\\%s1_%04d_pred.png"%("results",ind, file_name,i+ind*time_length)),np.uint8(output_image_color))
			cv2.imwrite(os.path.join(base_dir ,"%s\\%04d\\%s2_%04d_gt.png"%("results",ind, file_name,i+ind*time_length)),np.uint8(gt_i_color))
			cv2.imwrite(os.path.join(base_dir ,"%s\\%04d\\%s3_%04d_gt_img.png"%("results",ind, file_name,i+ind*time_length)),np.uint8(input_image_i))
			# cv2.imwrite(os.path.join(base_dir ,"%s\\%04d\\%s4_%04d_gt_normal.png"%("results",ind, file_name,i+ind*time_length)),np.uint8(gt_normals_i))
			# cv2.imwrite(os.path.join(base_dir ,"%s\\%04d\\%s5_%04d_pred_normal.png"%("results",ind, file_name,i+ind*time_length)),np.uint8(output_normals_i))
			# p = np.logical_and(output_image_i < 10 , output_image_i > -10)
			print('new')
			print(np.max(output_image_i), np.min(output_image_i))
			print(np.max(gt_i), np.min(gt_i))

			p =  output_image_i < 0.01
			g = gt_i < 0.01
			flat_pred = p.flatten()
			flat_label = g.flatten()
			score_averaging = 'weighted'
			prec = precision_score(flat_label, flat_pred, average=score_averaging)
			rec = recall_score(flat_label, flat_pred, average=score_averaging)
			f1 = f1_score(flat_label, flat_pred, average=score_averaging)
            #


			arg = confusion_matrix(flat_label, flat_pred).ravel()
			if (len(arg) != 1):
				tn, fp, fn, tp = arg
				accuracy = (tp + tn) / (tp + fp + fn + tn)
				iou =  tp / (tp + fp +fn)
			else:
				print("Not enough info")
				accuracy = np.nan
				iou =np.nan



			# Compute Precision-Recall and plot curve
			# precision, recall, thresholds = precision_recall_curve(flat_pred, flat_label )
			# area = auc(recall, precision)
			# print("Area Under Curve: %0.2f" % area)

			# plt.clf()
			# plt.plot(recall, precision, label='Precision-Recall curve')
			# plt.xlabel('Recall')
			# plt.ylabel('Precision')
			# plt.ylim([0.0, 1.05])
			# plt.xlim([0.0, 1.0])
			# plt.title('Precision-Recall example: AUC=%0.2f' % area)
			# plt.legend(loc="lower left")
			# plt.show()

			per_img_acc.append(accuracy)
			per_img_prec.append(prec)
			per_img_rec.append(rec)
			per_img_f1.append(f1)
			per_img_iou.append(iou)


			# g = np.expand_dims(gt_i,axis=2) == 0
			# p = np.uint8(np.where(p,[1,1,1],[0,0,0]))
			# g = np.uint8(np.where(g,[0,255,0],[0,0,0]))

			# cv2.addWeighted(p, 0.7, input_image_i, 0.3,0, input_image_i)


			# img_stacked = mask_color_img(input_image_i, p,  color = [255,0,0], alpha = 0.6)

			# img_stacked = utils.transparent_overlay(input_image_i, output_image_i)

			contour_lab = os.path.join(base_dir ,"%s\\%04d\\%s6_%04d_sub2.png"%("results",ind, file_name,i+ind*time_length))
			utils.interpolated_contour(input_image_i, output_image_i, contour_lab )
			# cv2.addWeighted(p, 0.5, g, 0.5,0, g)
			# cv2.imwrite(os.path.join(base_dir ,"%s\\%04d\\%s4_%04d_sub1.png"%("results",ind, file_name,i)),g)
	# 		cv2.imwrite(os.path.join(base_dir ,"%s\\%04d\\%s6_%04d_sub2.png"%("results",ind, file_name,i+ind*time_length)),np.uint8(img_stacked))
    #
    #
	# 		target.write("%f, %f, %f, %f, %f, %f\n"%(ind, accuracy, prec, rec, f1, iou))
	# 	target.close()
	# 	per_seq_acc.append(np.mean(per_img_acc))
	# 	per_seq_prec.append(np.mean(per_img_prec))
	# 	per_seq_rec.append(np.mean(per_img_rec))
	# 	per_seq_f1.append(np.mean(per_img_f1))
	# 	per_seq_iou.append(np.mean(per_img_iou))
    #
	# 	score.write("%f, %f, %f, %f, %f\n"%(np.nanmean(per_img_acc), np.nanmean(per_img_prec),\
    #                                         np.nanmean(per_img_rec), np.nanmean(per_img_f1),\
    #                                          np.nanmean(per_img_iou)))
	# score.close()





if __name__ == '__main__':
    main()
