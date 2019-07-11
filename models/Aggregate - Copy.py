"""
Temporal Pose
Common utility functions and classes.

Copyright (c) 2018 PCVLab & ADL
Licensed under the MIT License (see LICENSE for details)
Written by Nima A. Gard
"""


from __future__ import division
import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.keras as keras
import numpy as np



def block(inputs, n_filters, kernel_size=[2, 3, 3], strides = [1,1,1], down = True, dropout_p = 0.0, scope = 'block'):

	with tf.name_scope(scope):
		# print('**********  //-> {} // ***********'.format(scope))
		conv = tf.layers.conv3d(inputs, n_filters, kernel_size, strides =strides, activation=None , padding = 'same', use_bias=True, name = 'conv3d_%s_0'%scope)
		# conv = tf.nn.tanh(slim.batch_norm(conv, fused=True))
		# conv = tf.layers.conv3d(conv, n_filters, kernel_size, strides =strides, activation=None , padding = 'same', use_bias=True, name = 'conv3d_%s_1'%scope)
		# conv = tf.nn.tanh(slim.batch_norm(conv, fused=True))
		# conv = tf.layers.conv3d(conv, n_filters, kernel_size, strides =strides, activation=None , padding = 'same', use_bias=True, name = 'conv3d_%s_2'%scope)
		out = tf.nn.tanh(slim.batch_norm(conv, fused=True))
		if down:
			conv = tf.layers.conv3d(out, n_filters, kernel_size, strides =strides, activation=None , padding = 'same', use_bias=True, name = 'conv3d_%s_3'%scope)
			out = tf.nn.tanh(slim.batch_norm(conv, fused=True))

		if dropout_p != 0.0:
		  out = slim.dropout(out, keep_prob=(1.0-dropout_p))


	return out

def transpose_block(inputs, n_filters, kernel_size=[2, 2, 2], strides = [2, 2, 2],  dropout_p=0.0, scope = 'tBlock'):

	with tf.name_scope(scope):
		conv = tf.layers.conv3d_transpose(inputs, n_filters, kernel_size=kernel_size, strides = strides, use_bias= False, padding = 'SAME', name = 'tConv3d_%s_0'%scope)

	with tf.name_scope('activation_%s'%scope):
		out = tf.nn.tanh(slim.batch_norm(conv))
		if dropout_p != 0.0:
		  out = slim.dropout(out, keep_prob=(1.0-dropout_p))

	return out


def bulk(inputs, n_blocks, n_filters, kernel_size=[2,3,3], strides = [1,1,1], dropout_p=0.0, index = 0,	 scope = 'bulk'):
	filter_bag = [[1,3,3],[2,5,5],[2,7,7],[2,9,9],[2,11,11]]
	with tf.name_scope(scope):
		# print('**********  // {} // ***********'.format(scope))
		net = None
		for i in range(n_blocks):
			# print('**********  // {} // ***********'.format(i))
			bl = block(inputs, n_filters, filter_bag[i%5], strides, dropout_p=0.0, scope = 'bulk%02d_block%02d'%(index,i))

			if net is None:
				net = bl
			else:
				# net = tf.add(bl,net)#	tf.concat([net,bl],-1)
				net = tf.concat([net,bl],-1)
			# print('**********  {} ***********'.format(net.get_shape))
	return net

def transpose_bulk(inputs, n_blocks, n_filters, kernel_size=[2,2,2], strides = [2,2,2], dropout_p=0.0, index = 0, scope = 'tBulk'):

	with tf.name_scope(scope):
		# print('**********  // {} // ***********'.format(scope))
		net = None

		for i in range(n_blocks):
			bl = block(inputs, n_filters, down = False, scope = 'tbulk%02d_block%02d'%(index,i))
			bl = transpose_block(bl, n_filters, kernel_size, strides, dropout_p=0.0, scope = 'tBulk%02d_tBlock%02d'%(index,i))
			if net is None:
				net = bl
			else:
				net = tf.add(net,bl)
			# print('**********  {} ***********'.format(net.get_shape))
	return net

def build_Aggregate(inputs, num_classes, preset_model = "Aggregate", dropout_p=0.4, scope=None):
	"""
	Aggregate
	"""

	n_bulks = 5
	net = block(inputs, 64, kernel_size=[3, 3, 3], strides = [1,1,1], dropout_p = 0.0, scope = 'blockFirst')


	# print('**********  {} ***********'.format(net.get_shape))
	for i in range(n_bulks-1):
		net = bulk(net, 2**(n_bulks-i-1), 2**(2*i+2),index =  i, scope = 'bulk%02d'%i) ## with tf.concat High to low

		# net = bulk(net, 1, 2**(8+i),index =  i, scope = 'bulk%02d'%i) ## with tf.add

		with tf.name_scope('maxpool_%02d'%i):
			# print('**********  //-> {} // ***********'.format('maxpool_%02d'%i))
			net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],padding="SAME")
		# print('**********  {} ***********'.format(net.get_shape))
	first_tlayer = True
	for i in range(n_bulks - 2, -1, -1):
		if first_tlayer:
			net = transpose_bulk(net, 1, 2**(7+i), strides=[1,2,2], index = i + 1, scope = 'tBulk%02d'%(i+1))
			first_tlayer = False
		else:
			net = transpose_bulk(net, 1, 2**(6+i), index = i + 1, scope = 'tBulk%02d'%(i+1))
		# print('**********  {} ***********'.format(net.get_shape))


	final = slim.conv3d(net, num_classes, (1,1,1), activation_fn= tf.nn.tanh, scope='logits')
	return final, []
