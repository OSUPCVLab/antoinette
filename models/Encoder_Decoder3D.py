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


def conv_block(inputs, n_filters, kernel_size=(3, 3,3), strides = (1,1,1), dropout_p=0.0):
	"""
	Basic conv block for Encoder-Decoder
	Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
	Dropout (if dropout_p > 0) on the inputs
	"""
	conv = tf.layers.conv3d(inputs, n_filters, kernel_size, strides =strides, activation=None , padding = 'same')


	out = tf.nn.tanh(slim.batch_norm(conv, fused=True)) #changed relu to tanh
	if dropout_p != 0.0:
	  out = slim.dropout(out, keep_prob=(1.0-dropout_p))
	return out

def conv_transpose_block(inputs, n_filters, kernel_size=(2, 2, 2), strides = (2, 2,2),  dropout_p=0.0):
	"""
	Basic conv transpose block for Encoder-Decoder upsampling
	Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
	Dropout (if dropout_p > 0) on the inputs
	"""
	conv = tf.layers.conv3d_transpose(inputs, n_filters, kernel_size=kernel_size, strides = strides, use_bias=False, padding = 'SAME')

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
	n_filters = 32
	# Down 1
	contr_1_1 = conv_block(inputs, n_filters)
	contr_1_2  = conv_block(contr_1_1, n_filters)
	pool_1 = tf.nn.max_pool3d(contr_1_2 , ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],padding="SAME")

	# Down 2
	contr_2_1 = conv_block(pool_1, n_filters*2)
	contr_2_2 = conv_block(contr_2_1, n_filters*2)
	pool_2 = tf.nn.max_pool3d(contr_2_2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 1, 1],padding="SAME")

	# Down 3
	contr_3_1 = conv_block(pool_2, n_filters*4)
	contr_3_2 = conv_block(contr_3_1, n_filters*4)
	pool_3 = tf.nn.max_pool3d(contr_3_2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],padding="SAME")


	# Bottleneck
	pool_3 = slim.dropout(pool_3, keep_prob=(0.4))

	encode_1 = conv_block(pool_3, n_filters*4)
	encode_2 = conv_block(encode_1, n_filters*8)
	deconv_1 = conv_transpose_block(encode_2,  n_filters*4)

	#####################
	# Upsampling path #
	#####################

	concat2 = tf.concat([deconv_1, contr_3_2],1)

	# Up 1
	expand_2_1 = conv_block(concat2, n_filters*4, stride = 1)
	expand_2_2 = conv_block(expand_2_1, n_filters*4, stride = 1)
	deconv_3 = conv_transpose_block(expand_2_2, n_filters*4)

	output_2 = tf.layers.conv3d(deconv_3,num_classes,kernel_size=1, strides=1, padding='same', use_bias=True)
	output_2_up = keras.layers.UpSampling3D(size=(1, 2, 2))(output_2)

	concat3 = tf.concat([deconv_3, contr_2_2],1)

	# Up 2
	expand_3_1 = conv_block(concat3 , n_filters*2, strides = 1)
	expand_3_2 = conv_block(expand_3_1, n_filters*2, strides = 1)
	deconv_4 = conv_block(expand_3_2, n_filters)

	output_3 = tf.add(output_2_up, tf.layers.conv3d(concat3,num_classes,kernel_size=1, strides=1, padding='same', use_bias=True))
	output_3_up = keras.layers.UpSampling3D(size=(1, 2, 2))(output_3)


	# Up 3
	concat4 = tf.concat([deconv_4,contr_1_2],1)
	expand_4_1 = conv_block(concat4, n_filters, strides = 1)
	expand_4_2 = conv_block(expand_4_1, n_filters, strides = 1)

	conv_5 = tf.layers.conv3d(expand_4_2,num_classes,kernel_size=1, strides=1, padding='same', use_bias=True)

	final = tf.add(output_3_up, conv_5)
	net = slim.conv3d(final, 1, (1,1,1), activation_fn= tf.nn.tanh, scope='logits')

	return net
