import os
import time

from subprocess import call

import keras.backend as K
from keras.initializers import Ones, Zeros
from keras.models import Input, Model
from keras.layers import Conv2DTranspose
import tensorflow as tf
import scipy.misc
from PIL import Image

from tensorflow.python.framework import ops


import cv2
import matplotlib.pyplot as plt
import numpy as np


dataset_dir =  '../data/datasets/driving_dataset'
steer_image = '../data/.logo/steering_wheel_image.jpg'
steer_baseline = '../data/data.txt'
error_baseline = '../data/saliency_dataset/error_data.txt' 


def show_activation():
    """show the activations of the first two feature map layers"""

    # randomly choose an img from dataset  23000
    full_image = scipy.misc.imread(dataset_dir+ "/20000" + ".jpg", mode="RGB")
    # input planes: 3@66x200 & Normalization
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0 #  don't forget the grayscale

    

    saver = tf.train.Saver()

    # model has been constructed from import
    # with tf.Graph().as_default():

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        
        saver.restore(sess, model_dir)#Already trained model is loaded
        print("Load session successfully")

        conv1act, conv2act, conv3act, conv4act, conv5act , W_conv5_n , streeing_out = sess.run(#loading all teh convolutional layers
            fetches=[model_nvidia.h_conv1, model_nvidia.h_conv2, model_nvidia.h_conv3, model_nvidia.h_conv4, model_nvidia.h_conv5, model_nvidia.W_conv5 , model_nvidia.steering_output],  # 
            feed_dict={
                model_nvidia.x: [image],
                model_nvidia.keep_prob: 1.0
            }
        )

    
    conv1img = _generate_feature_image(conv1act[0], [6, int(conv1act.shape[3]/6)])

    fig = plt.figure('Visualization of Internal CNN State')
    plt.subplot(211)
    plt.title('Normalized input planes 3@66x200 to the CNN - Steer output: ' + str(streeing_out) + " -- Steer output: ")
    plt.imshow(image)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        # saver.restore(sess, model_dir)
        print("Starting deconv")

        # get the mean, and supress the first(batch) dimension
        averageC5_n = tf.reduce_mean(conv5act, 3)
        averageC5_n = tf.image.per_image_standardization(averageC5_n);
        averageC5_n = tf.expand_dims(averageC5_n, -1)

        averageC4_n = tf.reduce_mean(conv4act, 3)
        averageC4_n = tf.expand_dims(averageC4_n, -1)

        averageC3_n = tf.reduce_mean(conv3act, 3)
        averageC3_n = tf.expand_dims(averageC3_n, -1)
            
        averageC2_n = tf.reduce_mean(conv2act, 3)
        averageC2_n = tf.expand_dims(averageC2_n, -1)

        averageC1_n = tf.reduce_mean(conv1act, 3)
        averageC1_n = tf.expand_dims(averageC1_n, -1)

        W_conv5_av = weight_variable_vis([3, 3,1,1])
        W_conv4_av = weight_variable_vis([3, 3,1,1])
        W_conv3_av = weight_variable_vis([5, 5,1,1])
        W_conv2_av = weight_variable_vis([5, 5,1,1])
        W_conv1_av = weight_variable_vis([5, 5,1,1])

        sess.run(tf.global_variables_initializer())


        x = Input(shape=(None, None, 1))
        y1 = Conv2DTranspose(filters=1, 
                            kernel_size=(3,3), 
                            strides=(1,1), 
                            padding='valid', 
                            kernel_initializer=Ones(), 
                            bias_initializer=Zeros())(x)

        y2 = Conv2DTranspose(filters=1, 
                            kernel_size=(5,5), 
                            strides=(2,2), 
                            padding='valid', 
                            kernel_initializer=Ones(), 
                            bias_initializer=Zeros())(x)

        deconv_model_1 = Model(inputs=[x], outputs=[y1])
        deconv_model_2 = Model(inputs=[x], outputs=[y2])

        inps1 = [deconv_model_1.input, K.learning_phase()]   # input placeholder                                
        outs1 = [deconv_model_1.layers[-1].output]           # output placeholder
        deconv_func1 = K.function(inps1, outs1)              # evaluation function

        inps2 = [deconv_model_2.input, K.learning_phase()]   # input placeholder                                
        outs2 = [deconv_model_2.layers[-1].output]           # output placeholder
        deconv_func2 = K.function(inps2, outs2)              # evaluation function

        deconv5_avg = tf.nn.conv2d_transpose(value= averageC5_n, filter= W_conv5_av, output_shape=[1,3,20,1], strides=[1,1,1,1], padding='VALID');
        multC45 = tf.multiply(averageC4_n, deconv5_avg)  # use tensor multiplication.. not nm

        deconv4_avg = tf.nn.conv2d_transpose(value= multC45, filter= W_conv4_av, output_shape=[1,5,22,1], strides=[1,1,1,1], padding='VALID');
        multC34 = tf.multiply(averageC3_n, deconv4_avg)  

        deconv3_avg = tf.nn.conv2d_transpose(value= multC34, filter= W_conv3_av, output_shape=[1,14,47,1], strides=[1,2,2,1], padding='VALID');
        multC23 = tf.multiply(averageC2_n, deconv3_avg) 

        deconv2_avg = tf.nn.conv2d_transpose(value= multC23, filter= W_conv2_av, output_shape=[1,31,98,1], strides=[1,2,2,1], padding='VALID');
        multC12 = tf.multiply(averageC1_n, deconv2_avg)  

        the_mask = tf.nn.conv2d_transpose(value= multC12, filter= W_conv1_av, output_shape=[1,66,200,1], strides=[1,2,2,1], padding='VALID');
        test_mask = sess.run(the_mask)
        
        test_mask_img = np.mean(test_mask, axis=3).squeeze(axis=0)
        salient_mask = (test_mask_img - np.min(test_mask_img))/(np.max(test_mask_img) - np.min(test_mask_img) + 1e-6)
        salient_mask2 = (test_mask_img)/(np.max(test_mask_img) - np.min(test_mask_img) + 1e-6)
        
        deconv5_inter = sess.run(deconv4_avg) # ttdeconv5_avg)
        deconv4_inter = sess.run(deconv4_avg)
        deconv3_inter = sess.run(deconv3_avg)
        deconv2_inter = sess.run(deconv2_avg)
        deconv2_mask = np.mean(deconv2_inter, axis=3).squeeze(axis=0)
        deconv3_mask = np.mean(deconv3_inter, axis=3).squeeze(axis=0)
        deconv4_mask = np.mean(deconv4_inter, axis=3).squeeze(axis=0)
        deconv5_mask = np.mean(deconv5_inter, axis=3).squeeze(axis=0)
        
        act_map_5 = np.mean(sess.run(averageC5_n), axis=3).squeeze(axis=0)
        act_map_4 = np.mean(sess.run(averageC4_n), axis=3).squeeze(axis=0)
        act_map_3 = np.mean(sess.run(averageC3_n), axis=3).squeeze(axis=0)
        act_map_2 = np.mean(sess.run(averageC2_n), axis=3).squeeze(axis=0)
        act_map_1 = np.mean(sess.run(averageC1_n), axis=3).squeeze(axis=0)


        plt.subplot(212)
        plt.imshow(salient_mask, cmap='gray')
        plt.show()
        
        fig2 = plt.figure('Visualization of Activations vs. Intermediate Masks')
        plt.subplot(5,2,1)
        plt.imshow(act_map_1, cmap='gray')
        plt.subplot(5,2,3)
        plt.imshow(act_map_2, cmap='gray')
        plt.subplot(5,2,5)
        plt.imshow(act_map_3, cmap='gray')
        plt.subplot(5,2,7)
        plt.imshow(act_map_4, cmap='gray')
        plt.subplot(5,2,9)
        plt.imshow(act_map_5, cmap='gray')

        plt.subplot(5,2,2)
        plt.imshow(test_mask_img, cmap='gray')  # salient_mask salient_mask2
        plt.subplot(5,2,4)
        plt.imshow(deconv2_mask, cmap='gray')
        plt.subplot(5,2,6)
        plt.imshow(deconv3_mask, cmap='gray')
        plt.subplot(5,2,8)
        plt.imshow(deconv4_mask, cmap='gray')
        plt.subplot(5,2,10)
        plt.imshow(deconv5_mask, cmap='gray')

        plt.show()

        fig3 = plt.figure('Layer 1 Activations')
        plt.imshow(conv1img, cmap='gray')
        plt.show()
        

    cv2.destroyAllWindows()
    


## ------------ Backprop steps and details 
show_activation()    
