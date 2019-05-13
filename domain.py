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

EPOCHS = 10000
import logging
from contextlib import ExitStack
import tflearn
from tensorflow.contrib import slim
from collections import deque
from collections import OrderedDict
from tqdm import tqdm
def adversarial_discriminator(net, layers, scope='adversary', leaky=False, scope_resue=False):
    if leaky:
        activation_fn = tflearn.activations.leaky_relu
    else:
        activation_fn = tf.nn.relu
    with ExitStack() as stack:
        stack.enter_context(tf.variable_scope(scope,reuse=scope_resue))
        stack.enter_context(
            slim.arg_scope(
                [slim.fully_connected],
                activation_fn=activation_fn,
                weights_regularizer=slim.l2_regularizer(2.5e-5)))


        net = slim.conv3d(net, 1024, [3, 3, 3])
        net = slim.conv3d(net, 512, [3, 3, 3])
        net = slim.conv3d(net, 128, [3, 3, 3])
        net = slim.conv3d(net, 64, [3, 3, 3])
        net = slim.max_pool3d(net, [2, 2, 2])
        net = slim.flatten(net)
        # for dim in layers:
        #     net = slim.fully_connected(net, dim)
        net = slim.fully_connected(net, 2, activation_fn=None)

    return net
#
#
# def adversarial_discriminator_f(net, layers, scope='advf', leaky=False, scope_resue=False):
#     if leaky:
#         activation_fn = tflearn.activations.leaky_relu
#     else:
#         activation_fn = tf.nn.relu
#     with ExitStack() as stack:
#         stack.enter_context(tf.variable_scope(scope,reuse=scope_resue))
#         stack.enter_context(
#             slim.arg_scope(
#                 [slim.fully_connected],
#                 activation_fn=activation_fn,
#                 weights_regularizer=slim.l2_regularizer(2.5e-5)))
#
#
#         net = slim.conv3d(net, 1024, [3, 3, 3])
#         net = slim.conv3d(net, 512, [3, 3, 3])
#         net = slim.conv3d(net, 128, [3, 3, 3])
#         net = slim.conv3d(net, 64, [3, 3, 3])
#         net = slim.max_pool3d(net, [2, 2, 2])
#         net = slim.flatten(net)
#         # for dim in layers:
#         #     net = slim.fully_connected(net, dim)
#         net = slim.fully_connected(net, 2, activation_fn=None)

    return net


def main():
    adversary_layers = [500, 500]
    base_dir = os.getcwd()
    utils.config_logging()
    time_length = 8
    S_train_lab, S_val_lab, S_test_lab = utils.prepare_data_refresh(os.path.join(base_dir , "Data\\ReFresh"), time_length)
    T_train_lab, T_val_lab, T_test_lab = utils.prepare_data_posetrack(os.path.join(base_dir , "Data\\PoseTrack\\images"), time_length)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    num_classes =  1
    S_input =  tf.placeholder(tf.float32,shape=[1,time_length,128,128,3])
    T_input =  tf.placeholder(tf.float32,shape=[1,time_length,128,128,3])

    with tf.variable_scope('source'):
    	source_fts,source_fts_f = model_builder.build_model(model_name='UNet-3D',frontend ='ResNet101', net_input=S_input, num_classes=num_classes)
    with tf.variable_scope('target'):
    	target_fts,target_fts_f = model_builder.build_model(model_name='UNet-3D',frontend ='ResNet101', net_input=T_input, num_classes=num_classes)

    # adversarial network
    source_ft = source_fts#[:,3,:,:,:]
    target_ft = target_fts#[:,3,:,:,:]
    # source_ft = tf.reshape(source_ft, [-1, int(source_ft.get_shape()[-1])])
    # target_ft = tf.reshape(target_ft, [-1, int(target_ft.get_shape()[-1])])
    adversary_ft = tf.concat([source_ft, target_ft], 0)
    source_adversary_label = tf.ones([tf.shape(source_ft)[0]], tf.int32)
    target_adversary_label = tf.zeros([tf.shape(target_ft)[0]], tf.int32)
    adversary_label = tf.concat(
    	[source_adversary_label, target_adversary_label], 0)
    adversary_logits = adversarial_discriminator(
    	adversary_ft, adversary_layers, leaky=True    )
    # mapping_logits = adversarial_discriminator(
    # 	target_ft, adversary_layers, leaky=False , scope_resue = True )



	# adversary_ft_f = tf.concat([source_fts_f, target_fts_f], 0)
	# source_adversary_label_f = tf.zeros([tf.shape(source_fts_f)[0]], tf.int32)
	# target_adversary_label_f = tf.ones([tf.shape(target_fts_f)[0]], tf.int32)
	# adversary_label_f = tf.concat(
	# 	[source_adversary_label_f, target_adversary_label_f], 0)
	# mapping_logits_f = adversarial_discriminator_f(
	# 	target_fts_f, adversary_layers, leaky=False )

	# adversary_logits = tf.reshape(adversary_logits, [-1, int(adversary_logits.get_shape()[-1])])
	# variable collection
    source_vars = utils.collect_vars('source')
    target_vars = utils.collect_vars('target')
    adversary_vars = utils.collect_vars('adversary')
    # adversary_vars_f = utils.collect_vars('advf')
    # losses
    with tf.name_scope('mapping_loss'):
    	mapping_loss = tf.losses.sparse_softmax_cross_entropy(
    		1 - adversary_label, adversary_logits)
    tf.summary.scalar('mapping_loss', mapping_loss)
    with tf.name_scope('adversary_loss'):
    	adversary_loss = tf.losses.sparse_softmax_cross_entropy(
    		adversary_label, adversary_logits)
    tf.summary.scalar('adversary_loss', adversary_loss)
    # with tf.name_scope('loss_mappingf'):
    # 	   mapping_loss_f = tf.reduce_mean(tf.nn.l2_loss(source_fts_f - target_fts_f))
    # tf.summary.scalar('loss_mappingf', mapping_loss_f)
    #tf.losses.sparse_softmax_cross_entropy(target_adversary_label_f, mapping_logits_f)


    lr = 0.001
    lr_var = tf.Variable(lr, name = "learning_rate", trainable = False)
    optimizer = tf.train.AdamOptimizer(lr_var, 0.5)
    with tf.name_scope('mapping_step'):
    	mapping_step = optimizer.minimize(
    		mapping_loss, var_list=list(target_vars.values()))
    with tf.name_scope('adversary_step'):
    	adversary_step = optimizer.minimize(
    		adversary_loss, var_list=list(adversary_vars.values()))
    with tf.name_scope('gradsMapping'):
    	grads_mapping =  optimizer.compute_gradients(mapping_loss)
    with tf.name_scope('gradsAdversary'):
    	grads_adversary =  optimizer.compute_gradients(adversary_loss)

    # for g in grads_mapping:
    # 	tf.summary.histogram("%s_0-gradsM" % g[1].name[:-2], g[0])
    # for g in grads_adversary:
    # 	tf.summary.histogram("%s_0-gradA" % g[1].name[:-2], g[0])
    # with tf.name_scope('train_mappingf'):
    # 	mapping_step_f = optimizer.minimize(
    # 		mapping_loss_f, var_list= list(target_vars.values()))


    S_mean, S_std = 143.33426757268464, 17.119051447320118
    S_stacks_test = utils.Stacker(S_train_lab, time_length)
    print('Soruce statistics: {}, {}'.format(S_mean, S_std))
    T_stacks_test = utils.Stacker(T_train_lab, time_length)
    T_mean, T_std = T_stacks_test.preprocess()
    print('Target statistics: {}, {}'.format(T_mean, T_std))
    batch_size  = 1
    label_values = []

    sess.run(tf.global_variables_initializer())
    logging.info('    Restoring source model:')
    for src, tgt in source_vars.items():
    	logging.info('        {:30} -> {:30}'.format(src, tgt.name))
    source_restorer = tf.train.Saver(var_list=source_vars)
    model_checkpoint_name = os.path.join(base_dir , "checkpoints\\latest_model_EncoderDecoder.ckpt")
    source_restorer.restore(sess, model_checkpoint_name)

    logging.info('    Restoring target model:')
    for src, tgt in target_vars.items():
    	logging.info('        {:30} -> {:30}'.format(src, tgt.name))
    target_restorer = tf.train.Saver(var_list=target_vars)
    target_restorer.restore(sess, model_checkpoint_name)

    output_dir = os.path.join(base_dir,'snapshot')
    if not os.path.exists(output_dir):
    	os.mkdir(output_dir)

    bar  = tqdm(range(EPOCHS))
    bar.set_description('(lr: {:.0e})'.format(lr))
    bar.refresh()

    mapping_losses = deque(maxlen=200)
    adversary_losses = deque(maxlen=200)
    stepsize = None
    min_loss_mapping = 3
    min_loss_adversary = 3
    cric = 5

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(base_dir , "logs"), sess.graph)
    for i in bar:
    	batch_mapping_losses = []
    	batch_adversary_losses = []
    	total_number_of_batch = min(len(S_test_lab['input']), len(T_test_lab['input']))

    	for ind in range(0, total_number_of_batch):
    		S_gt, S_input_image = S_stacks_test.get_refresh_v3(ind,3)
    		S_input_image = np.expand_dims(S_input_image,axis=0)

    		_, T_input_image = T_stacks_test.get_posetrack(ind)
    		T_input_image = np.expand_dims(T_input_image,axis=0)
    		summary, mapping_loss_val, _= sess.run(
    			[merged, mapping_loss, mapping_step], feed_dict = {S_input: S_input_image,
    																T_input: T_input_image})
    		train_writer.add_summary(summary, ind + i * total_number_of_batch)
    		if ind % cric == 0:
    			adversary_loss_val, _ = sess.run(
    			[adversary_loss,adversary_step], feed_dict = {S_input: S_input_image,
    														   T_input: T_input_image})
    			batch_adversary_losses.append(adversary_loss_val)

    		# print(np.shape(source_out))
    		# plt.imshow(target_out[0,0,:,:,0])
    		# plt.show()
    		# print('*****************')
    		# print(logs)
    		# print('*****************')
    		batch_mapping_losses.append(mapping_loss_val)
    		# batch_adversary_losses.append(adversary_loss_val)

    	# if i % 1 == 0:
    	mapping_losses.append(np.mean(batch_mapping_losses))
    	adversary_losses.append(np.mean(batch_adversary_losses))
    	logging.info('{:20} Mapping: {:10.4f}     (avg: {:10.4f})'
    				'    Adversary: {:10.4f}     (avg: {:10.4f})'
    				.format('Iteration {}:'.format(i),
    						np.mean(batch_mapping_losses),
    						np.mean(mapping_losses),
    						np.mean(batch_adversary_losses),
    						np.mean(adversary_losses)))
    	if stepsize is not None and (i + 1) % stepsize == 0:
    		lr = sess.run(lr_var.assign(lr * 0.1))
    		logging.info('Changed learning rate to {:.0e}'.format(lr))
    		bar.set_description('{} (lr: {:.0e})'.format(output, lr))
    	if mapping_losses[-1] < min_loss_mapping and adversary_losses[-1] < min_loss_adversary :#and mapping_losses[-1] < 0.69 and adversary_losses[-1] < 0.69 :
    		min_loss_mapping = mapping_losses[-1]
    		min_loss_adversary = adversary_losses[-1]
    		snapshot_path = target_restorer.save(
    			sess,os.path.join(base_dir,'snapshot\\latest_model_EncoderDecoder.ckpt'))
    		logging.info('Saved snapshot to {}'.format(snapshot_path))

    sess.close()
    train_writer.close()

if __name__ == '__main__':
    main()
