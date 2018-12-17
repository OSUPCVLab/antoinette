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
import numpy as np


def conv_block(inputs, n_filters, kernel_size=[3, 3,3], dropout_p=0.0):
	"""
	Basic conv block for Encoder-Decoder
	Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
	Dropout (if dropout_p > 0) on the inputs
	"""
	conv = slim.conv3d(inputs, n_filters, (1,1,1), activation_fn=None, normalizer_fn=None , padding =  'SAME')
	conv = slim.conv3d(conv, n_filters, (kernel_size[0],1,1), activation_fn=None, normalizer_fn=None, padding =  'SAME')
	conv = slim.conv3d(conv, n_filters, (1,kernel_size[1],kernel_size[2]), activation_fn=None, normalizer_fn=None, padding =  'SAME')

	out = tf.nn.tanh(slim.batch_norm(conv, fused=True)) #changed relu to tanh
	if dropout_p != 0.0:
	  out = slim.dropout(out, keep_prob=(1.0-dropout_p))
	return out

def conv_transpose_block(inputs, n_filters, kernel_size=(3, 3, 3), dropout_p=0.0):
	"""
	Basic conv transpose block for Encoder-Decoder upsampling
	Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
	Dropout (if dropout_p > 0) on the inputs
	"""
	conv = tf.layers.conv3d_transpose(inputs, n_filters, kernel_size=(kernel_size[0], kernel_size[1],kernel_size[2]), strides=(2, 2,2),use_bias=False, padding = 'SAME')

	# conv = tf.nn.conv3d_transpose(inputs, filter = [3,3,3,inputs.shape[4],n_filters],output_shape = [-1,n_filters,n_filters,n_filters], strides=[1,2,2,2,1], padding = 'SAME')
	out = tf.nn.tanh(slim.batch_norm(conv))#changed relu to tanh, LeakyReLU
	if dropout_p != 0.0:
	  out = slim.dropout(out, keep_prob=(1.0-dropout_p))
	return out

def build_encoder_decoder_3d(inputs, num_classes, preset_model = "Encoder-Decoder-3D", dropout_p=0.5, scope=None):
	"""
	Builds the Encoder-Decoder model. Inspired by SegNet with some modifications
	Optionally includes skip connections

	Arguments:
	  inputs: the input tensor
	  n_classes: number of classes
	  dropout_p: dropout rate applied after each convolution (0. for not using)

	Returns:
	  Encoder-Decoder model
	"""


	if preset_model == "Encoder-Decoder-3D":
		has_skip = False
	elif preset_model == "Encoder-Decoder-Skip-3D":
		has_skip = True
	else:
		raise ValueError("Unsupported Encoder-Decoder model '%s'. This function only supports Encoder-Decoder and Encoder-Decoder-Skip" % (preset_model))

	#####################
	# Downsampling path #
	#####################
	net = conv_block(inputs, 64)
	net = conv_block(net, 64)
	net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 1, 2, 2, 1],padding="SAME")
	skip_1 = net

	net = conv_block(net, 128)
	net = conv_block(net, 128)
	net = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 2, 2, 1],padding="SAME")
	skip_2 = net

	net = conv_block(net, 256)
	net = conv_block(net, 256)
	net = conv_block(net, 256)
	net = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1],padding="SAME")
	skip_3 = net

	net = conv_block(net, 512)
	net = conv_block(net, 512)
	net = conv_block(net, 512)
	net = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 2,2, 2, 1],padding="SAME")
	skip_4 = net

	net = conv_block(net, 512)
	net = conv_block(net, 512)
	net = conv_block(net, 512)
	net = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding = "SAME")

	#####################
	# Upsampling path #
	#####################

	net = conv_transpose_block(net, 512)
	net = conv_block(net, 512)
	net = conv_block(net, 512)
	net = conv_block(net, 512)
	if has_skip:
		net = tf.add(net, skip_4)

	net = conv_transpose_block(net, 512)
	net = conv_block(net, 512)
	net = conv_block(net, 512)
	net = conv_block(net, 256)
	if has_skip:
		net = tf.add(net, skip_3)
	net = conv_transpose_block(net, 256)
	net = conv_block(net, 256)
	net = conv_block(net, 256)
	net = conv_block(net, 128)
	if has_skip:
		net = tf.add(net, skip_2)

	net = conv_transpose_block(net, 128)
	net = conv_block(net, 128)
	net = conv_block(net, 64)
	net = tf.nn.max_pool3d(net, ksize=[1, 3, 1, 1, 1], strides=[1, 2, 1, 1, 1], padding = "SAME")
	if has_skip:
		net = tf.add(net, skip_1)


	net = conv_transpose_block(net, 64)
	net = conv_block(net, 64)
	net = conv_block(net, 64)

	#####################
	#      Softmax      #
	#####################

	net = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 1, 1, 1], padding = "SAME")
	net = slim.conv3d(net, 1, (1,1,1), activation_fn= tf.nn.tanh, scope='logits')
	return net
