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



def block(inputs, n_filters, kernel_size=(3, 3, 3), strides = (1,1,1), dropout_p = 0.0, scope = 'block'):

	with tf.name_scope(scope):
		# print('**********  //-> {} // ***********'.format(scope))
		conv = tf.layers.conv3d(inputs, n_filters, kernel_size, strides =strides, activation=None , padding = 'same')
		conv = tf.layers.conv3d(conv, n_filters, kernel_size, strides =strides, activation=None , padding = 'same')
		conv = tf.layers.conv3d(conv, n_filters, kernel_size, strides =strides, activation=None , padding = 'same')
		conv = tf.layers.conv3d(conv, n_filters, kernel_size, strides =strides, activation=None , padding = 'same')

	with tf.name_scope('activation_%s'%scope):
		# print('**********  //-> {} // ***********'.format('activation_%s'%scope))
		out = tf.nn.tanh(slim.batch_norm(conv, fused=True))
		if dropout_p != 0.0:
		  out = slim.dropout(out, keep_prob=(1.0-dropout_p))


	return out

def transpose_block(inputs, n_filters, kernel_size=(2, 2, 2), strides = (2, 2, 2),  dropout_p=0.0, scope = 'transposeBlock'):

	with tf.name_scope(scope):
		conv = tf.layers.conv3d_transpose(inputs, n_filters, kernel_size=kernel_size, strides = strides, use_bias=False, padding = 'SAME')

	with tf.name_scope('activation_%s'%scope):
		out = tf.nn.tanh(slim.batch_norm(conv))
		if dropout_p != 0.0:
		  out = slim.dropout(out, keep_prob=(1.0-dropout_p))

	return out


def bulk(inputs, n_blocks, n_filters, kernel_size=(3,3,3), strides = (1,1,1), dropout_p=0.0, scope = 'bulk'):

	with tf.name_scope(scope):
		# print('**********  // {} // ***********'.format(scope))
		net = None
		for i in range(n_blocks):
			# print('**********  // {} // ***********'.format(i))
			bl = block(inputs, n_filters, kernel_size, strides, dropout_p=0.0, scope = 'block%02d'%i)
			if net is None:
				net = bl
			else:
				net = tf.concat([net,bl],-1)
			# print('**********  {} ***********'.format(net.get_shape))
	return net

def transpose_bulk(inputs, n_blocks, n_filters, kernel_size=(1,2,2), strides = (1,2,2), dropout_p=0.0, scope = 'transposeBulk'):

	with tf.name_scope(scope):
		# print('**********  // {} // ***********'.format(scope))
		net = None
		for i in range(n_blocks):
			bl = transpose_block(inputs, n_filters, kernel_size, strides, dropout_p=0.0, scope = 'transposeBlock%02d'%i)
			if net is None:
				net = bl
			else:
				net = tf.concat([net,bl],-1)
			# print('**********  {} ***********'.format(net.get_shape))
	return net

def build_Aggregate(inputs, num_classes, preset_model = "Aggregate", dropout_p=0.4, scope=None):
	"""
	Aggregate
	"""

	n_bulks = 6
	net = tf.layers.conv3d(inputs, 2,kernel_size=(2, 2, 2), strides = (2, 2, 2),  activation=None , padding = 'same')

	# print('**********  {} ***********'.format(net.get_shape))
	for i in range(n_bulks-1):
		net = bulk(net, 2**(n_bulks - i), 2**(2*i+1), scope = 'bulk%02d'%i)
		with tf.name_scope('maxpool_%02d'%i):
			# print('**********  //-> {} // ***********'.format('maxpool_%02d'%i))
			net = tf.nn.max_pool3d(net, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1],padding="SAME")
		# print('**********  {} ***********'.format(net.get_shape))

	for i in range(n_bulks - 2, -1, -1):
		net = transpose_bulk(net, 2**(n_bulks - i), 2**(2*i+1), scope = 'transposeBulk%02d'%(i+1))
		# print('**********  {} ***********'.format(net.get_shape))
	net = transpose_block(net, 64, kernel_size=(2, 2, 2), strides = (2, 2, 2),  dropout_p=0.0, scope = 'transposeBlock00')

	final = slim.conv3d(net, num_classes, (1,1,1), activation_fn= tf.nn.tanh, scope='logits')

	return final, []
