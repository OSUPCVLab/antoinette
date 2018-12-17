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

EPOCHS = 10

def main():
    time_length = 20
    train_lab, val_lab, test_lab = utils.prepare_data('./Data')
    filenames_train,annotations_train = utils.generate_labels_with_permutation(train_lab, time_length)
    filenames_val,annotations_val = utils.generate_labels_with_permutation(val_lab, time_length)
    config = tf.ConfigProto()
    sess = tf.Session(config = config)
    num_classes =  128
    net_input =  tf.placeholder(tf.float32,shape=[None,None,None,3])
    net_output = tf.placeholder(tf.float32,shape=[None,None,None,1])#,None,num_classes

    network = model_builder.build_model(model_name='Encoder-Decoder',frontend ='ResNet101', net_input=net_input, num_classes=num_classes)
    loss = tf.reduce_mean(tf.nn.l2_loss(network- net_output))#softmax_cross_entropy_with_logits_v2(logits = network, labels = net_output)
    learning_rate_list = np.array(np.random.uniform(-6, 0,20))
    lr_loss = np.zeros(len(learning_rate_list))
    for lr_ind in range(len(learning_rate_list)):

        print(learning_rate_list[lr_ind])
        opt = tf.train.RMSPropOptimizer(learning_rate = 10 ** learning_rate_list[lr_ind], decay = 0.995).minimize(loss, var_list = [var for var in tf.trainable_variables()])

        saver = tf.train.Saver(max_to_keep  = 1000)
        sess.run(tf.global_variables_initializer())

        model_checkpoint_name = "checkpoints/latest_model_EncoderDecoder.ckpt"

        avg_loss_per_epoch = []
        avg_scores_per_epoch = []
        avg_iou_per_epoch = []

        # Which validation images do we want
        val_indices = []
        num_vals = min(20, len(filenames_val))
        random.seed(16)
        np.random.seed(16)
        #filenames_val=random.sample(filenames_val,num_vals)
        val_indices=random.sample(range(0,len(filenames_val)),num_vals)

        stacks_train = utils.Stacker(annotations_train, time_length)
        stacks_val = utils.Stacker(annotations_val, time_length)
        batch_size  = 1
        label_values = []
        #label_values.append([255,0,0])
        #label_values.append([0,255,0])

        label_values = utils.get_spaced_colors(128)

        for epoch in range(EPOCHS):
            current_losses = []
            cnt = 0
            st = time.time()
            epoch_st=time.time()
            filenames_train = np.random.permutation(filenames_train)
            num_iters = int(np.floor(len(filenames_train) / batch_size))
            for i in range(num_iters):
                # st=time.time()

                input_image_batch = []
                output_image_batch = []

                # Collect a batch of images
                for j in range(batch_size):
                    index = i* batch_size + j

                    # print(filenames_train[index])
                    img_output, img_input = stacks_train.vectorize_stack_images_2d_temporal(filenames_train[index])
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
                    output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))
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
            if not os.path.isdir("%s/%04d"%("checkpoints",epoch)):
                os.makedirs("%s/%04d"%("checkpoints",epoch))

            # Save latest checkpoint to same file name
            print("Saving latest checkpoint")
            saver.save(sess,model_checkpoint_name)

            if val_indices != 0 and epoch % 5 == 0:
                print("Saving checkpoint for this epoch")
                saver.save(sess,"%s/%04d/model.ckpt"%("checkpoints",epoch))


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
                    gt, input_image = stacks_val.vectorize_stack_images_2d_temporal(filenames_val[ind])
                    input_image = np.expand_dims(input_image,axis=0)
                    # gt = helpers.reverse_one_hot(helpers.onehot(gt))
                    output_image = sess.run(network, feed_dict = {net_input: input_image})
                    output_image = np.array(output_image[0,:,:,:])
                    # output_image = helpers.reverse_one_hot(output_image)

                    gt = gt[time_length-1,:].reshape((64,64))
                    output_image = output_image[time_length-1,:].reshape((64,64))*255.0


                    output_image = utils.image_serialization([64,64,1], output_image)
                    gt = utils.image_serialization([64,64,1], gt)
                    input_image = input_image[0,time_length-1,:,:].reshape(64,64,3)
                    input_image = utils.image_serialization([64,64,3], input_image)
                    # out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
                    # output_viz_image1, output_viz_image2 = utils.inverse_distance_transform(gt,output_image)

                    # accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)

                    file_name = utils.filepath_to_name(filenames_val[ind][0])


                    # scores_list.append(accuracy)
                    # class_scores_list.append(class_accuracies)
                    # precision_list.append(prec)
                    # recall_list.append(rec)
                    # f1_list.append(f1)
                    # iou_list.append(iou)

                    gt = helpers.colour_code_segmentation(gt, label_values)

                    file_name = os.path.basename(filenames_val[ind][0])
                    file_name = os.path.splitext(file_name)[0]
                    cv2.imwrite("%s/%04d/%s_pred.png"%("checkpoints",epoch, file_name),np.uint8(output_image))#cv2.cvtColor(np.uint8(), cv2.COLOR_RGB2BGR))
                    cv2.imwrite("%s/%04d/%s_gt.png"%("checkpoints",epoch, file_name),np.uint8(gt))#cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))
                    cv2.imwrite("%s/%04d/%s_gt_img.png"%("checkpoints",epoch, file_name),np.uint8(input_image))#cv2.cvtColor(np.uint8(input_image[0,time_length-1,:,:].reshape(64,64,3)), cv2.COLOR_RGB2BGR))



                # avg_score = np.mean(scores_list)
                # class_avg_scores = np.mean(class_scores_list, axis=0)
                # avg_scores_per_epoch.append(avg_score)
                # avg_precision = np.mean(precision_list)
                # avg_recall = np.mean(recall_list)
                # avg_f1 = np.mean(f1_list)
                # avg_iou = np.mean(iou_list)
                # avg_iou_per_epoch.append(avg_iou)

                # print("\nAverage validation accuracy for epoch # %04d = %f"% (epoch, avg_score))
                # print("Average per class validation accuracies for epoch # %04d:"% (epoch))
                # #for index, item in enumerate(class_avg_scores):
                # #    print("%s = %f" % (class_names_list[index], item))
                # print("Validation precision = ", avg_precision)
                # print("Validation recall = ", avg_recall)
                # print("Validation F1 score = ", avg_f1)
                # print("Validation IoU score = ", avg_iou)

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

            #
            # fig1, ax1 = plt.subplots(figsize=(11, 8))
            #
            # ax1.plot(range(epoch+1), avg_scores_per_epoch)
            # ax1.set_title("Average validation accuracy vs epochs")
            # ax1.set_xlabel("Epoch")
            # ax1.set_ylabel("Avg. val. accuracy")
            #
            #
            # plt.savefig('accuracy_vs_epochs.png')
            #
            # plt.clf()
            #
            fig2, ax2 = plt.subplots(figsize=(11, 8))

            ax2.plot(range(epoch+1), avg_loss_per_epoch)
            ax2.set_title("Average loss vs epochs")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Current loss")

            plt.savefig('loss_vs_epochs_{0}.png'.format(learning_rate_list[lr_ind]))

            plt.clf()
            #
            # fig3, ax3 = plt.subplots(figsize=(11, 8))
            #
            # ax3.plot(range(epoch+1), avg_iou_per_epoch)
            # ax3.set_title("Average IoU vs epochs")
            # ax3.set_xlabel("Epoch")
            # ax3.set_ylabel("Current IoU")
            #
            # plt.savefig('iou_vs_epochs.png')

        lr_loss[lr_ind] = np.mean(avg_loss_per_epoch)
        fig3, ax3 = plt.subplots(figsize=(11, 8))
        tmp_lr_list = np.array(learning_rate_list[0:lr_ind+1])
        tmp_lr_ind_sorted = tmp_lr_list.argsort()
        ax3.plot(tmp_lr_list[tmp_lr_ind_sorted], lr_loss[tmp_lr_ind_sorted])
        ax3.set_title("Average loss vs Learning Rate")
        ax3.set_xlabel("learning rate")
        ax3.set_ylabel("Current loss")

        plt.savefig('loss_vs_lr.png')

        plt.clf()
        #
if __name__ == '__main__':
    main()
