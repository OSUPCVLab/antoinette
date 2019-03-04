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
import ntpath
from tensorflow.python.framework import ops
# use 'Agg' on matplotlib so that plots could be generated even without Xserver
# running
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
EPOCHS = 1000

def implicit(vol):
  eps = 1e-15
  img_output_3d = ndimage.distance_transform_edt(vol)
  inside = vol == 0.0
  temp = ndimage.distance_transform_edt(1 - vol)
  img_output_3d = np.where(inside,np.maximum(-temp,-10), np.minimum(img_output_3d,10))
  img_output_3d = (img_output_3d - (np.min(img_output_3d))) / (np.max(img_output_3d) - np.min(img_output_3d)+ eps)

  return img_output_3d

def compute_implicit_and_normals(data):
  # compute implicit surface (vol), then calculate normals
  impl = implicit(data)
  gradients = np.gradients(impl)

  return impl, gradients

def combined_loss(y1,y2, sess):
	impl1, grads1 = compute_implicit_and_normals(y1.eval(session = sess))
	impl2, grads2 = compute_implicit_and_normals(y2.eval(session = sess))

	loss_l2 = tf.reduce_mean(tf.nn.l2_loss(tf.convert_to_tensor(impl1 - impl2, np.float32)))
	loss_normals = tf.reduce_mean(tf.nn.l2_loss(tf.convert_to_tensor(grads1 - grads2, np.float32)))


	return loss_l2 + loss_normals


def main():
	base_dir = os.getcwd()

	time_length = 16
	# train_lab, val_lab, test_lab = utils.prepare_data(os.path.join(base_dir , "Data\\SURREAL"), time_length,mode = 'TRAIN')
	train_lab, val_lab, test_lab = utils.prepare_data_refresh(os.path.join(base_dir , "Data\\ReFresh"), time_length)

	print('Number of training images: {}'.format(len(train_lab['input'])))
	print('Number of validation images: {}'.format(len(val_lab['input'])))
	print('Number of test images: {}'.format(len(test_lab['input'])))

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config = config)
	num_classes =  1
	net_input =  tf.placeholder(tf.float32,shape=[None,time_length,None,None,3])
	net_output = tf.placeholder(tf.float32,shape=[None,time_length,None,None,1])#,None,num_classes

	network = model_builder.build_model(model_name='UNet-3D',frontend ='ResNet101', net_input=net_input, num_classes=num_classes)
	# loss = tf.reduce_mean(tf.nn.l2_loss(network- net_output))#softmax_cross_entropy_with_logits_v2(logits = network, labels = net_output)
	# loss = tf.reduce_mean(tf.losses.huber_loss(network,net_output))
	# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = network,labels =net_output))
	loss = combined_loss(network,  net_output, sess)

	# np.savetxt('sd',output_derivatives, delimiter = ',' )
	# loss_norms =
	# loss_l2 =
	# loss = lo
	# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = network,labels =net_output))
	# opt = tf.train.AdamOptimizer(learning_rate = 0.001 ).minimize(loss, var_list = [var for var in tf.trainable_variables()])

	## New AdamOptimizer
	global_step = tf.Variable(0, name='global_step', trainable=False)
	learning_rate = tf.train.exponential_decay(0.0001, global_step, EPOCHS, 0.9, staircase=True)
	opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

	saver = tf.train.Saver(max_to_keep  = 1000)
	sess.run(tf.global_variables_initializer())

	model_checkpoint_name = os.path.join(base_dir , "checkpoints\\latest_model_EncoderDecoder.ckpt")
	# saver.restore(sess, model_checkpoint_name)
	avg_loss_per_epoch = []
	avg_scores_per_epoch = []
	avg_iou_per_epoch = []

	# Which validation images do we want
	val_indices = []
	num_vals = 20
	random.seed(16)
	np.random.seed(16)

	val_indices=random.sample(range(0,len(val_lab['input'])),num_vals)

	stacks_train = utils.Stacker(train_lab, time_length)
	stacks_val = utils.Stacker(val_lab, time_length)

	batch_size  = 2
	label_values = []
	with sess.as_default():
		start = global_step.eval()
		for epoch in range(EPOCHS):
			current_losses = []
			cnt = 0
			st = time.time()
			epoch_st=time.time()

			num_iters = range(int(np.floor(len(stacks_train.info['input']) / batch_size)))
			num_iters = np.random.permutation(num_iters)
			for i in num_iters:
				# st=time.time()

				input_image_batch = []
				output_image_batch = []

				# Collect a batch of images
				for j in range(batch_size):
					index = i* batch_size + j
					img_output, img_input = stacks_train.get_refresh(index)

					with tf.device('/cpu:0'):

						img_input = np.float32(img_input) / 255.0
						input_image_batch.append(np.expand_dims(img_input, axis = 0))

						# img_output = np.float32(helpers.onehot(img_output))
						img_output = np.float32(np.expand_dims(img_output, axis = 3))

						output_image_batch.append(np.expand_dims(img_output, axis = 0))


				if batch_size == 1:
					input_image_batch = input_image_batch[0]
					output_image_batch = output_image_batch[0]
				else:
					input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
					output_image_batch = np.stack(output_image_batch, axis=1)[0]
				_, current = sess.run([opt, loss],
									feed_dict = {net_input : input_image_batch,
												 net_output: output_image_batch})
				current_losses.append(current)
				cnt += 1
				if cnt % 20 == 0:
					string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f"%(epoch,cnt,current,time.time()-st)
					utils.LOG(string_print)
					st = time.time()
			mean_loss = np.mean(current_losses)
			avg_loss_per_epoch.append(mean_loss)
			# Create directories if needed
			if not os.path.isdir(os.path.join(base_dir , "%s\\%04d"%("checkpoints",epoch))):
				os.makedirs(os.path.join(base_dir , "%s\\%04d"%("checkpoints",epoch)))

			# Save latest checkpoint to same file name
			print("Saving latest checkpoint")
			saver.save(sess,model_checkpoint_name)
			global_step.assign(epoch).eval()

			if val_indices != 0 and epoch % 5 == 0:
				print("Saving checkpoint for this epoch")
				saver.save(sess,  os.path.join(base_dir , "%s\\%04d\\model.ckpt"%("checkpoints",epoch)))


			if epoch % 1 == 0:
				print("Performing validation")

				scores_list = []
				class_scores_list = []
				precision_list = []
				recall_list = []
				f1_list = []
				iou_list = []


				# Do the validation on a small set of validation images
				for ind in val_indices:
					gt, input_image = stacks_val.get_refresh(ind)
					input_image = np.expand_dims(input_image,axis=0)
					output_image = sess.run(network, feed_dict = {net_input: input_image/255.0})
					output_image = np.array(output_image[0,:,:,:])

					gt = gt[time_length-1,:,:]*255.0
					output_image = output_image[time_length-1,:,:]*255.0

					input_image = input_image[0,time_length-1,:,:,:]

					# file_name = utils.filepath_to_name(filenames_val[ind][0])


					file_name = os.path.basename(val_lab['input'][ind][0])
					file_name = os.path.splitext(file_name)[0]
					cv2.imwrite(os.path.join(base_dir , "%s\\%04d\\%s_%04d_pred.png"%("checkpoints",epoch, file_name,ind)),np.uint8(output_image))
					cv2.imwrite(os.path.join(base_dir ,"%s\\%04d\\%s_%04d_gt.png"%("checkpoints",epoch, file_name,ind)),np.uint8(gt))
					cv2.imwrite(os.path.join(base_dir ,"%s\\%04d\\%s_%04d_gt_img.png"%("checkpoints",epoch, file_name,ind)),np.uint8(input_image))

			epoch_time=time.time()-epoch_st
			remain_time=epoch_time*(EPOCHS-1-epoch)
			m, s = divmod(remain_time, 60)
			h, m = divmod(m, 60)
			if s!=0:
				train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
			else:
				train_time="Remaining training time : Training completed.\n"
			utils.LOG(train_time)
			scores_list = []


			fig2, ax2 = plt.subplots(figsize=(11, 8))

			ax2.plot(range(epoch+1), avg_loss_per_epoch)
			ax2.set_title("Average loss vs epochs")
			ax2.set_xlabel("Epoch")
			ax2.set_ylabel("Current loss")

			plt.savefig(os.path.join(base_dir , 'loss_vs_epochs.png'))

			plt.clf()



if __name__ == '__main__':
    main()
