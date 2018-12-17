from __future__ import division
import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def conv_block(inputs, n_filters, kernel_size=[3, 3], dropout_p=0.0):
	"""
	Basic conv block for Encoder-Decoder
	Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
	Dropout (if dropout_p > 0) on the inputs
	"""
	conv = slim.conv2d(inputs, n_filters, kernel_size, activation_fn=None, normalizer_fn=None)
	out = tf.nn.relu(slim.batch_norm(conv, fused=True))
	if dropout_p != 0.0:
	  out = slim.dropout(out, keep_prob=(1.0-dropout_p))
	return out

def conv_transpose_block(inputs, n_filters, kernel_size=[3, 3], dropout_p=0.0):
	"""
	Basic conv transpose block for Encoder-Decoder upsampling
	Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
	Dropout (if dropout_p > 0) on the inputs
	"""
	conv = slim.conv2d_transpose(inputs, n_filters, kernel_size=[3, 3], stride=[2, 2], activation_fn=None)
	out = tf.nn.relu(slim.batch_norm(conv))
	if dropout_p != 0.0:
	  out = slim.dropout(out, keep_prob=(1.0-dropout_p))
	return out

def build_siamese_encoder_decoder(inputs, num_classes, preset_model = "Siamese-Encoder-Decoder", dropout_p=0.5, scope=None):
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


	if preset_model == "Siamese-Encoder-Decoder":
		has_skip = False
	elif preset_model == "Siamese-Encoder-Decoder-Skip":
		has_skip = True
	else:
		raise ValueError("Unsupported Encoder-Decoder model '%s'. This function only supports Encoder-Decoder and Encoder-Decoder-Skip" % (preset_model))
	inputs1 = inputs[0]
	inputs2 = inputs[1]
	#####################
	# Downsampling path #
	#####################
	net1 = conv_block(inputs1, 64)
	net1 = conv_block(net1, 64)
	net1 = slim.pool(net1, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_1_1 = net1

	net2 = conv_block(inputs2, 64)
	net2 = conv_block(net2, 64)
	net2 = slim.pool(net2, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_2_1 = net2

	net_temp = net1
	net1 = tf.add(net1, net2)
	net2 = tf.add(net2, net_temp)

	# Layer 2
	net1 = conv_block(net1, 128)
	net1 = conv_block(net1, 128)
	net1 = slim.pool(net1, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_1_2 = net1

	net2 = conv_block(net2, 128)
	net2 = conv_block(net2, 128)
	net2 = slim.pool(net2, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_2_2 = net2

	net_temp = net1
	net1 = tf.add(net1, net2)
	net2 = tf.add(net2, net_temp)

	# Layer 3
	net1 = conv_block(net1, 256)
	net1 = conv_block(net1, 256)
	net1 = conv_block(net1, 256)
	net1 = slim.pool(net1, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_1_3 = net1

	net2 = conv_block(net2, 256)
	net2 = conv_block(net2, 256)
	net2 = conv_block(net2, 256)
	net2 = slim.pool(net2, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_2_3 = net2

	net_temp = net1
	net1 = tf.add(net1, net2)
	net2 = tf.add(net2, net_temp)

	# Layer 4
	net1 = conv_block(net1, 512)
	net1 = conv_block(net1, 512)
	net1 = conv_block(net1, 512)
	net1 = slim.pool(net1, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_1_4 = net1

	net2 = conv_block(net2, 512)
	net2 = conv_block(net2, 512)
	net2 = conv_block(net2, 512)
	net2 = slim.pool(net2, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_2_4 = net2

	net_temp = net1
	net1 = tf.add(net1, net2)
	net2 = tf.add(net2, net_temp)

	# Layer 5
	net1 = conv_block(net1, 512)
	net1 = conv_block(net1, 512)
	net1 = conv_block(net1, 512)
	net1 = slim.pool(net1, [2, 2], stride=[2, 2], pooling_type='MAX')

	net2 = conv_block(net2, 512)
	net2 = conv_block(net2, 512)
	net2 = conv_block(net2, 512)
	net2 = slim.pool(net2, [2, 2], stride=[2, 2], pooling_type='MAX')

	net_temp = net1
	net1 = tf.add(net1, net2)
	net2 = tf.add(net2, net_temp)


	#####################
	# Upsampling path #
	#####################
	# Layer 1
	net1 = conv_transpose_block(net1, 512)
	net1 = conv_block(net1, 512)
	net1 = conv_block(net1, 512)
	net1 = conv_block(net1, 512)
	if has_skip:
		net1 = tf.add(net1, skip_1_4)

	net2 = conv_transpose_block(net2, 512)
	net2 = conv_block(net2, 512)
	net2 = conv_block(net2, 512)
	net2 = conv_block(net2, 512)
	if has_skip:
		net2 = tf.add(net2, skip_2_4)

	net_temp = net1
	net1 = tf.add(net1, net2)
	net2 = tf.add(net2, net_temp)

	# Layer 2
	net1 = conv_transpose_block(net1, 512)
	net1 = conv_block(net1, 512)
	net1 = conv_block(net1, 512)
	net1 = conv_block(net1, 256)
	if has_skip:
		net1 = tf.add(net1, skip_1_3)

	net2 = conv_transpose_block(net2, 512)
	net2 = conv_block(net2, 512)
	net2 = conv_block(net2, 512)
	net2 = conv_block(net2, 256)
	if has_skip:
		net2 = tf.add(net2, skip_2_3)

	net_temp = net1
	net1 = tf.add(net1, net2)
	net2 = tf.add(net2, net_temp)

	# Layer 3
	net1 = conv_transpose_block(net1, 256)
	net1 = conv_block(net1, 256)
	net1 = conv_block(net1, 256)
	net1 = conv_block(net1, 128)
	if has_skip:
		net1 = tf.add(net1, skip_1_2)

	net2 = conv_transpose_block(net2, 256)
	net2 = conv_block(net2, 256)
	net2 = conv_block(net2, 256)
	net2 = conv_block(net2, 128)
	if has_skip:
		net2 = tf.add(net2, skip_2_2)

	net_temp = net1
	net1 = tf.add(net1, net2)
	net2 = tf.add(net2, net_temp)

	# Layer 4
	net1 = conv_transpose_block(net1, 128)
	net1 = conv_block(net1, 128)
	net1 = conv_block(net1, 64)
	if has_skip:
		net1 = tf.add(net1, skip_1_1)

	net2 = conv_transpose_block(net2, 128)
	net2 = conv_block(net2, 128)
	net2 = conv_block(net2, 64)
	if has_skip:
		net2 = tf.add(net2, skip_2_1)

	net_temp = net1
	net1 = tf.add(net1, net2)
	net2 = tf.add(net2, net_temp)

	# Layer 5
	net1 = conv_transpose_block(net1, 64)
	net1 = conv_block(net1, 64)
	net1 = conv_block(net1, 64)

	net2 = conv_transpose_block(net2, 64)
	net2 = conv_block(net2, 64)
	net2 = conv_block(net2, 64)


	net =  tf.add(net1, net2)


	#####################
	#      Softmax      #
	#####################
	# net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
	# net = tf.layers.dense(inputs=net, units  = 64* 64)
	# net = conv_block(net, 32)
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
	net = conv_transpose_block(net, 1)
	return net
