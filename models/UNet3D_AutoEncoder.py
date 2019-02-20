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
from tensorflow import distributions as ds
tfd = tf.contrib.distributions

import numpy as np


def conv_block(inputs, n_filters, kernel_size=(3, 3,3), strides = (1,1,1), dropout_p=0.0):
	"""
	Basic conv block for Encoder-Decoder
	Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
	Dropout (if dropout_p > 0) on the inputs
	"""
	conv = tf.layers.conv3d(inputs, n_filters, kernel_size, strides =strides, activation=None , padding = 'same')


	out = tf.nn.relu(slim.batch_norm(conv, fused=True)) #changed relu to tanh
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
	out = tf.nn.relu(slim.batch_norm(conv))#changed relu to tanh, LeakyReLU
	if dropout_p != 0.0:
	  out = slim.dropout(out, keep_prob=(1.0-dropout_p))
	return out

def build_UNet_3d_auto(inputs, num_classes, preset_model = "UNet-3D", dropout_p=0.5, scope=None):
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



	#####################
	# Downsampling path #
	#####################
	n_filters = 64
	# Down 1
	contr_1_1 = conv_block(inputs, n_filters)
	contr_1_2  = conv_block(contr_1_1, n_filters)
	pool_1 = tf.nn.max_pool3d(contr_1_2 , ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],padding="SAME")
	# Down 2
	contr_2_1 = conv_block(pool_1, n_filters*2)
	contr_2_2 = conv_block(contr_2_1, n_filters*2)
	pool_2 = tf.nn.max_pool3d(contr_2_2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],padding="SAME")
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
	#####################
	# 1. Implicit Surface #
	#####################
	# concat2 = tf.concat([deconv_1, contr_3_2],4)
	# # Up 1
	# expand_2_1 = conv_block(concat2, n_filters*4)
	# expand_2_2 = conv_block(expand_2_1, n_filters*4)
	# deconv_3 = conv_transpose_block(expand_2_2, n_filters*2)
	#
	# output_2 = tf.layers.conv3d(concat2,num_classes,kernel_size=1, strides=1, padding='same', use_bias=True)
	# output_2_up = keras.layers.UpSampling3D(size=(2, 2, 2))(output_2)
	# # ch, cw = self.get_crop_shape(conv3, output_2_up)
	# # output_2_up = layers.Cropping2D(cropping=(ch,cw))(output_2_up)
	# concat3 = tf.concat([deconv_3, contr_2_2],4)
	#
	# # Up 2
	# expand_3_1 = conv_block(concat3 , n_filters*2)
	# expand_3_2 = conv_block(expand_3_1, n_filters*2)
	# deconv_4 = conv_transpose_block(expand_3_2, n_filters)
	#
	# output_3 = tf.add(output_2_up, tf.layers.conv3d(concat3,num_classes,kernel_size=1, strides=1, padding='same', use_bias=True))
	#
	# output_3_up = keras.layers.UpSampling3D(size=(2, 2, 2))(output_3)
	#
	#
	# # Up 3
	# concat4 = tf.concat([deconv_4,contr_1_2],4)
	#
	# expand_4_1 = conv_block(concat4, n_filters)
	# expand_4_2 = conv_block(expand_4_1, n_filters)
	#
	# conv_5 = tf.layers.conv3d(expand_4_2,num_classes,kernel_size=1, strides=1, padding='same', use_bias=True)
	#
	# final = tf.add(output_3_up, conv_5)
	#
	# net1 = slim.conv3d(final, num_classes, (1,1,1), activation_fn=None, scope='logits')


	#####################
	# 2. Reconstruct Image#
	#####################
	h_flat = tf.reshape(deconv_1,[-1,n_filters*4*32*32*4])
	# print('h_flat: ', h_flat.get_shape())
	recognition = tf.layers.dense(h_flat,  n_filters*2, tf.relu)

	z_mean = tf.layers.dense(recognition, n_filters)

	z_stddev = tf.layers.dense(recognition, n_filters, tf.nn.softplus) + 1e-8#
	samples = tf.random_normal([1,n_filters],0,1,dtype=tf.float32)

	guessed_z = z_mean + (z_stddev * samples)

	generation = tf.layers.dense(guessed_z,n_filters*32*32*2 )
	generation = tf.reshape(generation,[-1,4,32,32,32])

	# Up 1
	a_expand_2_1 = conv_block(generation, n_filters*4)
	a_expand_2_2 = conv_block(a_expand_2_1, n_filters*4)
	a_deconv_3 = conv_transpose_block(a_expand_2_2, n_filters*2)

	# Up 2
	a_expand_3_1 = conv_block(a_deconv_3 , n_filters*2)
	a_expand_3_2 = conv_block(a_expand_3_1, n_filters*2)
	a_deconv_4 = conv_transpose_block(a_expand_3_2, n_filters)

	# Up 3
	a_expand_4_1 = conv_block(a_deconv_4, n_filters)
	a_expand_4_2 = conv_block(a_expand_4_1, n_filters)

	net2 = slim.conv3d(a_expand_4_2, 3, (1,1,1), activation_fn=None, scope='generation')
	net2 = tf.nn.sigmoid(net2)
	# result = tf.concat([net2,net1],4)s
	result = net2

	latent_loss =tf.reduce_mean( 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1))
	return result, latent_loss




def make_encoder(inputs, code_size):
	n_filters = 64
	# Down 1
	contr_1_1 = conv_block(inputs, n_filters)
	contr_1_2  = conv_block(contr_1_1, n_filters)
	pool_1 = tf.nn.max_pool3d(contr_1_2 , ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],padding="SAME")
	# Down 2
	contr_2_1 = conv_block(pool_1, n_filters*2)
	contr_2_2 = conv_block(contr_2_1, n_filters*2)
	pool_2 = tf.nn.max_pool3d(contr_2_2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],padding="SAME")
	# Down 3
	contr_3_1 = conv_block(pool_2, n_filters*4)
	contr_3_2 = conv_block(contr_3_1, n_filters*4)
	pool_3 = tf.nn.max_pool3d(contr_3_2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],padding="SAME")

	# Bottleneck
	pool_3 = slim.dropout(pool_3, keep_prob=(0.4))

	encode_1 = conv_block(pool_3, n_filters*4)
	encode_2 = conv_block(encode_1, n_filters*8)
	deconv_1 = conv_transpose_block(encode_2,  n_filters*4)

	x = tf.layers.flatten(deconv_1)
	x = tf.layers.dense(x, 200, tf.nn.relu)
	x = tf.layers.dense(x, 200, tf.nn.relu)
	loc = tf.layers.dense(x, code_size)
	scale = tf.layers.dense(x, code_size, tf.nn.softplus)
	return tfd.MultivariateNormalDiag(loc, scale)

def make_prior(code_size):
	loc = tf.zeros(code_size)
	scale = tf.ones(code_size)
	return tfd.MultivariateNormalDiag(loc, scale)

def make_decoder(code, data_shape):
	x = code
	x = tf.layers.dense(x, 200, tf.nn.relu)
	x = tf.layers.dense(x, 200, tf.nn.relu)
	logit = tf.layers.dense(x, np.prod(data_shape))
	logit = tf.reshape(logit, [-1] + data_shape)
	a_expand_2_1 = conv_block(logit, n_filters*4)
	a_expand_2_2 = conv_block(a_expand_2_1, n_filters*4)
	a_deconv_3 = conv_transpose_block(a_expand_2_2, n_filters*2)

	# Up 2
	a_expand_3_1 = conv_block(a_deconv_3 , n_filters*2)
	a_expand_3_2 = conv_block(a_expand_3_1, n_filters*2)
	a_deconv_4 = conv_transpose_block(a_expand_3_2, n_filters)

	# Up 3
	a_expand_4_1 = conv_block(a_deconv_4, n_filters)
	a_expand_4_2 = conv_block(a_expand_4_1, n_filters)

	net2 = slim.conv3d(a_expand_4_2, 3, (1,1,1), activation_fn=None, scope='generation')


	return net2
  	# return tfd.Independent(tfd.Bernoulli(logit), 3)
