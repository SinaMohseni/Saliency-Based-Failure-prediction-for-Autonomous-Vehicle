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


def run_model(network,starting_frame):
    
    if network == '5_layer_128_rgb':
        from nets import model_nvidia_128_rgb as model_nvidia
        model_dir = './logs/checkpoint_5_layer_lrg4/checkpoint/model.ckpt'  
        img_lrg = True
    elif network == '5_layer_128_gray':
        from nets.pilotNet_5_layer2 import PilotNet
        model_dir = './logs/checkpoint_5_layer_lrg6/checkpoint/model.ckpt'   
        img_lrg = True
    elif network == '5_layer_66_rgb':  
        from nets import model_nvidia
        model_dir = './logs/checkpoint_5_layer_sml/model.ckpt'  
        img_lrg = False
    elif network == '5_layer_66_gray':  
        from nets import model_nvidia_5_layer_66_gray as model_nvidia 
        model_dir = './logs/student_1/model_2.ckpt'  
        img_lrg = False
    


    f = open(steer_baseline, 'r')
    baseline_raw = f.read().splitlines()
    baseline = [x.split(' ')[1] for x in baseline_raw]
    f.close()


    WIN_MARGIN_LEFT = 240;
    WIN_MARGIN_TOP = 240;
    WIN_MARGIN_BETWEEN = 180;
    WIN_WIDTH = 480;

    model = PilotNet()

    saver = tf.train.Saver()

    img = cv2.imread(steer_image, 0)
    rows,cols = img.shape

    # Visualization init
    cv2.namedWindow("Steering Wheel", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Steering Wheel", WIN_MARGIN_LEFT, WIN_MARGIN_TOP)
    cv2.namedWindow("Scenario", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Scenario", WIN_MARGIN_LEFT+cols+WIN_MARGIN_BETWEEN, WIN_MARGIN_TOP)

    smoothed_angle = 0
    
    i = starting_frame


    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        
        saver.restore(sess, model_dir)
        print("Load session successfully")

      
        while(cv2.waitKey(10) != ord('q')):
                
            
            if (img_lrg == True):
                if (network == '5_layer_128_rgb'):
                    full_image = scipy.misc.imread(dataset_dir + "/" + str(i) + ".jpg",mode="RGB") # 
                    image = scipy.misc.imresize(full_image[-150:], [102, 364]) / 255.0

                elif (network == '5_layer_128_gray'):
                    full_image_1 = scipy.misc.imread(dataset_dir + "/" + str(i) + ".jpg",mode="RGB");  #, flatten=True
                    full_image = scipy.misc.imread(dataset_dir + "/" + str(i) + ".jpg", flatten =True )[-128:] / 255.0;  #, flatten=True
                    image = scipy.misc.imresize(full_image,[102, 364])
                    image = np.expand_dims(image, axis=-1);                    
            else:
                full_image = scipy.misc.imread(dataset_dir + "/" + str(i) + ".jpg",mode="RGB" ) # mode="RGB")
                image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0

            
            print("image.shape: ",image.shape)

            steering = sess.run(
                fetches=[model.steering],
                feed_dict={
                    model.image_input: [image],
                    model.keep_prob: 1.0
                }
            )

            degrees = float(steering[0][0] * 180.0 )/ scipy.pi
            call("clear")

            call("clear")
            print("Predicted SWA: ", degrees)
            print("Smoothed SWA: ", smoothed_angle)
            print("Baseline SWA: " , baseline[i])
            print("SWA Error: " + str(degrees - float(baseline[i])))

            # convert RGB due to dataset format
            cv2.imshow("Scenario", cv2.cvtColor(full_image_1, cv2.COLOR_RGB2BGR))
            cv2.imshow("Model input", image) 

            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
            M = cv2.getRotationMatrix2D((cols/2,rows/2), -smoothed_angle, 1)
            dst = cv2.warpAffine(img,M,(cols,rows))
            cv2.imshow("Steering Wheel", dst)

        
            i += 1


    cv2.destroyAllWindows()


## ------------ runs the PilotNet model to see the output  
run_model('5_layer_128_gray', starting_frame= 20000)     # 5_layer_66_rgb   5_layer_128_rgb  5_layer_128_gray
