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
from utils import utils, helpers
from builders import model_builder
import os

# use 'Agg' on matplotlib so that plots could be generated even without Xserver
# running
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EPOCHS = 1000

def main():
    base_dir = os.getcwd()
    time_length = 16
    train_lab, val_lab, test_lab = utils.prepare_data(os.path.join(base_dir ,'Data\\PoseTrack'))

    filenames_test,annotations_test = utils.generate_labels_with_permutation(val_lab, time_length)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    num_classes =  1
    net_input =  tf.placeholder(tf.float32,shape=[None,time_length,None,None,3])
    net_output = tf.placeholder(tf.float32,shape=[None,time_length,None,None,1])#,None,num_classes

    network = model_builder.build_model(model_name='Encoder-Decoder-Skip-3D',frontend ='ResNet101', net_input=net_input, num_classes=num_classes)
    loss = tf.reduce_mean(tf.nn.l2_loss(network- net_output))#softmax_cross_entropy_with_logits_v2(logits = network, labels = net_output)




    opt = tf.train.AdamOptimizer(learning_rate = 0.001 ).minimize(loss, var_list = [var for var in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep  = 1000)
    sess.run(tf.global_variables_initializer())

    model_checkpoint_name = os.path.join(base_dir , "checkpoints\\latest_model_EncoderDecoder.ckpt")
    saver.restore(sess, model_checkpoint_name)



    stacks_test = utils.Stacker(annotations_test, time_length)
    batch_size  = 1
    label_values = []

    for ind in range(0,len(filenames_test)):
            gt, input_image = stacks_test.vectorize_stack_images_3d_temporal(filenames_test[ind])
            input_image = np.expand_dims(input_image,axis=0)
            output_image = sess.run(network, feed_dict = {net_input: input_image/255.0})
            output_image = np.array(output_image[0,:,:,:])
            if not os.path.isdir(os.path.join(base_dir , "%s\\%04d"%("results2",ind))):
                os.makedirs(os.path.join(base_dir , "%s\\%04d"%("results2",ind)))
            for i in range(time_length):
                gt_i = gt[i,:,:]*255.0
                output_image_i = output_image[i,:,:]*255.0

                input_image_i = input_image[0,i,:,:,:]

                file_name = utils.filepath_to_name(filenames_test[ind][0])


                file_name = os.path.basename(filenames_test[ind][0])
                file_name = os.path.splitext(file_name)[0]

                cv2.imwrite(os.path.join(base_dir ,"%s\\%04d\\%s1_%04d_pred.png"%("results2",ind, file_name,i)),np.uint8(output_image_i))
                cv2.imwrite(os.path.join(base_dir ,"%s\\%04d\\%s2_%04d_gt.png"%("results2",ind, file_name,i)),np.uint8(gt_i))
                cv2.imwrite(os.path.join(base_dir ,"%s\\%04d\\%s3_%04d_gt_img.png"%("results2",ind, file_name,i)),np.uint8(input_image_i))
                p = output_image_i < 20
                g = np.expand_dims(gt_i,axis=2) == 0
                p = np.uint8(np.where(p,[255,0,0],[0,0,0]))
                g = np.uint8(np.where(g,[0,255,0],[0,0,0]))

                cv2.addWeighted(p, 0.7, input_image_i, 0.3,0, input_image_i)
                cv2.addWeighted(p, 0.5, g, 0.5,0, g)
                cv2.imwrite(os.path.join(base_dir ,"%s\\%04d\\%s4_%04d_sub1.png"%("results2",ind, file_name,i)),g)
                cv2.imwrite(os.path.join(base_dir ,"%s\\%04d\\%s5_%04d_sub2.png"%("results2",ind, file_name,i)),np.uint8(input_image_i))






if __name__ == '__main__':
    main()
