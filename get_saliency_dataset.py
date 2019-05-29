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


def weight_variable_vis(shape):
    initial = tf.constant(1.0, shape=shape)
    return tf.Variable(initial)

def _generate_feature_image(feature_map, shape):
    dim = feature_map.shape[2];
    row_step = feature_map.shape[0];
    col_step = feature_map.shape[1];

    feature_image = np.zeros([row_step*shape[0], col_step*shape[1]]);
    min = np.min(feature_map);
    max = np.max(feature_map);
    minmax = np.fabs(min - max);
    cnt = 0;
    for row in range(shape[0]):
        row_idx = row_step * row;
        row_idx_nxt = row_step * (row + 1);
        for col in range(shape[1]):
            col_idx = col_step * col;
            col_idx_nxt = col_step * (col + 1);
            feature_image[row_idx:row_idx_nxt, col_idx:col_idx_nxt] = (feature_map[:, :, cnt] - min) * 1.0/minmax;
            cnt += 1;
    return feature_image;


def visual_back_prop(network,starting_frame):
    
    if network == '5_layer_128_rgb':
        # from nets import model_nvidia_128_rgb as model_nvidia
        from nets.pilotNet_5_layer_128_rgb import PilotNet
        model_dir = './logs/checkpoint_5_layer_lrg4/checkpoint/model.ckpt'  
        
        img_lrg = True
    elif network == '5_layer_66_rgb':
        # from nets import model_nvidia
        from nets.pilotNet_5_layer_new import PilotNet
        model_dir = './logs/new_model_org/logs_orig_model/checkpoint/model.ckpt'    # ./logs/checkpoint_5_layer_sml/model.ckpt
        img_lrg = False

   	f = open(steer_baseline, 'r')
    baseline_raw = f.read().splitlines()
    baseline = [x.split(' ')[1] for x in baseline_raw]
    f.close()

    WIN_MARGIN_LEFT = 240;
    WIN_MARGIN_TOP = 240;
    WIN_MARGIN_BETWEEN = 180;
    WIN_WIDTH = 480;

    img = cv2.imread(steer_image, 0)
    rows,cols = img.shape

    cv2.namedWindow("Steering Wheel", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Steering Wheel", WIN_MARGIN_LEFT, WIN_MARGIN_TOP)

    smoothed_angle = 0

    model_nvidia = PilotNet()

    saver = tf.train.Saver()

    
    i = starting_frame   # ----------   start from frame = 3000 ---------------

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        
        saver.restore(sess, model_dir)    
        print("Load session successfully")

        while(cv2.waitKey(10) != ord('q')):  
           	if (img_lrg == True):
                full_image = scipy.misc.imread(dataset_dir + "/" + str(i) + ".jpg")
                image = scipy.misc.imread(dataset_dir + "/" + str(i) + ".jpg", mode="RGB")[-128:] / 255.0;  
                image = scipy.misc.imresize(image,[102, 364])
            else:
                full_image = scipy.misc.imread(dataset_dir + "/" + str(i) + ".jpg",mode="RGB" ) 
                image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0

            conv1act, conv2act, conv3act, conv4act, conv5act , streeing_out_1 = sess.run(
                fetches=[model_nvidia.h_conv1, model_nvidia.h_conv2, model_nvidia.h_conv3, model_nvidia.h_conv4, model_nvidia.h_conv5 , model_nvidia.steering_output],  # 
                feed_dict={
                    model_nvidia.x: [image],
                    model_nvidia.keep_prob: 1.0
                }
            )


            
            averageC5_n = tf.reduce_mean(conv5act, 3)

            averageC5_n = tf.reduce_mean(conv5act, 3)
            averageC5_n = tf.expand_dims(averageC5_n, -1)
            averageC4_n = tf.reduce_mean(conv4act, 3)
            averageC4_n = tf.expand_dims(averageC4_n, -1)

            averageC3_n = tf.reduce_mean(conv3act, 3)
            averageC3_n = tf.expand_dims(averageC3_n, -1)
                
            averageC2_n = tf.reduce_mean(conv2act, 3)
            averageC2_n = tf.expand_dims(averageC2_n, -1)

            averageC1_n = tf.reduce_mean(conv1act, 3)
            averageC1_n = tf.image.per_image_standardization(averageC1_n);
            averageC1_n = tf.expand_dims(averageC1_n, -1)

            W_conv5_av = weight_variable_vis([3, 3,1,1])
            W_conv4_av = weight_variable_vis([3, 3,1,1])
            W_conv3_av = weight_variable_vis([5, 5,1,1])
            W_conv2_av = weight_variable_vis([5, 5,1,1])
            W_conv1_av = weight_variable_vis([5, 5,1,1])
            
            sess.run(W_conv5_av.initializer)
            sess.run(W_conv4_av.initializer)
            sess.run(W_conv3_av.initializer)
            sess.run(W_conv2_av.initializer)
            sess.run(W_conv1_av.initializer)

            print('---------------- ',sess.run(tf.report_uninitialized_variables()))


            if (img_lrg == True):

                deconv5_avg = tf.nn.conv2d_transpose(value= averageC5_n, filter= W_conv5_av, output_shape=[1,8,40,1], strides=[1,1,1,1], padding='VALID');
                multC45 = tf.multiply(averageC4_n, deconv5_avg)  

                deconv4_avg = tf.nn.conv2d_transpose(value= multC45, filter= W_conv4_av, output_shape=[1,10,42,1], strides=[1,1,1,1], padding='VALID');
                multC34 = tf.multiply(averageC3_n, deconv4_avg)  

                deconv3_avg = tf.nn.conv2d_transpose(value= multC34, filter= W_conv3_av, output_shape=[1,23,88,1], strides=[1,2,2,1], padding='VALID');
                multC23 = tf.multiply(averageC2_n, deconv3_avg)  

                deconv2_avg = tf.nn.conv2d_transpose(value= multC23, filter= W_conv2_av, output_shape=[1,49,180,1], strides=[1,2,2,1], padding='VALID');
                multC12 = tf.multiply(averageC1_n, deconv2_avg)  

                the_mask = tf.nn.conv2d_transpose(value= multC12, filter= W_conv1_av, output_shape=[1,102,364,1], strides=[1,2,2,1], padding='VALID');
                sess.run(tf.global_variables_initializer())
                test_mask = sess.run(the_mask)
            else: 
                deconv5_avg = tf.nn.conv2d_transpose(value= averageC5_n, filter= W_conv5_av, output_shape=[1,3,20,1], strides=[1,1,1,1], padding='VALID');
                multC45 = tf.multiply(averageC4_n, deconv5_avg)  

                deconv4_avg = tf.nn.conv2d_transpose(value= multC45, filter= W_conv4_av, output_shape=[1,5,22,1], strides=[1,1,1,1], padding='VALID');
                multC34 = tf.multiply(averageC3_n, deconv4_avg)  

                deconv3_avg = tf.nn.conv2d_transpose(value= multC34, filter= W_conv3_av, output_shape=[1,14,47,1], strides=[1,2,2,1], padding='VALID');
                multC23 = tf.multiply(averageC2_n, deconv3_avg)  

                deconv2_avg = tf.nn.conv2d_transpose(value= multC23, filter= W_conv2_av, output_shape=[1,31,98,1], strides=[1,2,2,1], padding='VALID');
                multC12 = tf.multiply(averageC1_n, deconv2_avg)  

                the_mask = tf.nn.conv2d_transpose(value= multC12, filter= W_conv1_av, output_shape=[1,66,200,1], strides=[1,2,2,1], padding='VALID');

            
            test_mask_out = sess.run(the_mask)

            test_mask_img = np.mean(test_mask_out, axis=3).squeeze(axis=0)
            salient_mask = (test_mask_img - np.min(test_mask_img))/(np.max(test_mask_img) - np.min(test_mask_img) + 1e-6)
            salient_mask2 = (test_mask_img)/(np.max(test_mask_img) - np.min(test_mask_img) + 1e-6)
                
            

            degrees_1 = streeing_out_1[0][0] * 180.0 / scipy.pi;
            streeing_out= streeing_out_1
            degrees = degrees_1; 
            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
            
            call("clear")
            print("Predicted SWA: ", streeing_out , degrees, degrees_1) 
            print("Smoothed SWA: " + str(smoothed_angle))
            print("Baseline SWA: " + baseline[i])
            print("SWA Error: " + str(degrees - float(baseline[i])))

            if (i == starting_frame):
                fig = plt.figure()
                plt.subplot(211)
                plt.title('Camera Input and Saliency Map')
                camera_obj = plt.imshow(image)
                plt.subplot(212)
                saliency_obj = plt.imshow(salient_mask2) # , cmap='gray'

            camera_obj.set_data(image)
            saliency_obj.set_data(salient_mask2)

            
            M = cv2.getRotationMatrix2D((cols/2,rows/2), -smoothed_angle, 1)
            dst = cv2.warpAffine(img,M,(cols,rows))
            cv2.imshow("Steering Wheel", dst)

            fig.canvas.draw_idle()
            plt.pause(0.001)
            i+=1


    print ("writng in txt file...")
    txt_file = open('../'+model_dir+'/err_data.txt', 'w')
    i=0
    while(i <total_frames):
        img_name = str(i) + ".jpg"
        txt_file.write(img_name + " " +str(steering_error_arr[i])+ "\n")
        i+=1
    txt_file.close()     



def visual_back_prop2(network,starting_frame,last_frame):
        
    ops.reset_default_graph()

    if network == '5_layer_128_rgb':
        from nets import model_nvidia_128_rgb as model_nvidia
        # from nets import deconv_128_rgb as deconv
        model_dir = './logs/checkpoint_5_layer_lrg3/checkpoint/model.ckpt'    # checkpoint_5_layer_lrg2-sina-RGB
        img_lrg = True;
        rgb = True;
    elif network == '5_layer_66_rgb':
        from nets import model_nvidia
        # from nets import deconv_66_rgb as deconv
        model_dir = './logs/new_model_org/logs_orig_model/checkpoint/model.ckpt'
                    #'./logs/checkpoint_5_layer_sml/model.ckpt'  
        img_lrg = False;
        rgb = True;
    elif network == '5_layer_128_gray':
        from nets import model_nvidia_128_gray  as model_nvidia
        # from nets.pilotNet_5_layer2 import PilotNet
        model_dir = './logs/checkpoint_5_layer_lrg4/checkpoint/model.ckpt'  
        img_lrg = True;
        rgb = False;
        


    f = open(steer_baseline, 'r')
    baseline_raw = f.read().splitlines()
    baseline = [x.split(' ')[1] for x in baseline_raw]
    f.close()

    WIN_MARGIN_LEFT = 240;
    WIN_MARGIN_TOP = 240;
    WIN_MARGIN_BETWEEN = 180;
    WIN_WIDTH = 480;
    # model = PilotNet()
    img = cv2.imread(steer_image, 0)
    rows,cols = img.shape

    cv2.namedWindow("Steering Wheel", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Steering Wheel", WIN_MARGIN_LEFT, WIN_MARGIN_TOP)

    smoothed_angle = 0

    [act_list_1, act_list_2,act_list_3,act_list_4,act_list_5, error_out, streeing_out] = get_activations(False, model_nvidia,img_lrg,rgb,model_dir,starting_frame, last_frame, baseline)
    
    salient_mask_out = get_saliency(False, act_list_1, act_list_2,act_list_3,act_list_4,act_list_5, 
                                    model_nvidia, img_lrg,rgb,model_dir,starting_frame,(last_frame - starting_frame))

  


def visual_back_prop3(teacher_network,starting_frame,last_frame,baseline,smoothed_angle):

    ops.reset_default_graph()

    if teacher_network == '5_layer_128_rgb':
        from nets import model_nvidia_128_rgb as model_nvidia
        # from nets import deconv_128_rgb as deconv
        model_dir = './logs/checkpoint_5_layer_lrg3/checkpoint/model.ckpt'    # checkpoint_5_layer_lrg2-sina-RGB
        img_lrg = True;
        rgb = True;
    elif teacher_network == '5_layer_66_rgb':
        from nets import model_nvidia as teacher_network
        from nets import model_nvidia as student_network
        # from nets import deconv_66_rgb as deconv
        model_dir = './logs/new_model_org/logs_orig_model/checkpoint/model.ckpt'
                    #'./logs/checkpoint_5_layer_sml/model.ckpt'  
        img_lrg = False;
        rgb = True;
    elif teacher_network == '5_layer_128_gray':
        from nets import model_nvidia_128_gray  as model_nvidia
        model_dir = './logs/checkpoint_5_layer_lrg-akshay-gray/checkpoint/model5layer.ckpt'  
        img_lrg = True;
        rgb = False;
        
    # print (teacher_network, student_network),

    
    [act_list_1, act_list_2,act_list_3,act_list_4,act_list_5, streeing_out] = get_activations(False, teacher_network,img_lrg,rgb,model_dir,starting_frame, last_frame, baseline)
    
    salient_mask_out = get_saliency(False, act_list_1, act_list_2,act_list_3,act_list_4,act_list_5, 
                                    teacher_network, img_lrg,rgb,model_dir,starting_frame,(last_frame - starting_frame))



    WIN_MARGIN_LEFT = 240;
    WIN_MARGIN_TOP = 240;
    WIN_MARGIN_BETWEEN = 180;
    WIN_WIDTH = 480;

    img_steering = cv2.imread(steer_image, 0)


    rows,cols = img_steering.shape

    cv2.namedWindow("Steering Wheel", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Steering Wheel", WIN_MARGIN_LEFT, WIN_MARGIN_TOP)
    M = cv2.getRotationMatrix2D((cols/2,rows/2), 0, 1)
    dst = cv2.warpAffine(img_steering,M,(cols,rows))
    cv2.imshow("Steering Wheel", dst)
    
    for i in range(0, last_frame-starting_frame-3): 

        degrees = streeing_out[i]
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)

        call("clear")
        print("Predicted SWA: ", degrees)
        print("Smoothed SWA: ", smoothed_angle)
        print("Baseline SWA: " , baseline[i])
        print("SWA Error: " + str(degrees - float(baseline[i])))


        if (i == 0): #starting_frame):
            fig = plt.figure()
            plt.subplot(211)
            plt.title('Camera Input and Saliency Map')
            # if (rgb ==False): 
            #     camera_obj = plt.imshow(camera_image, cmap='gray')
            # else: 
            #     camera_obj = plt.imshow(camera_image)

            plt.subplot(212)
            saliency_obj = plt.imshow(salient_mask_out[i][0],cmap='gray', vmin=0, vmax=1)

        # camera_obj.set_data(camera_image)
        # print (starting_frame + i)
        saliency_obj.set_data(salient_mask_out[i][0])

        M = cv2.getRotationMatrix2D((cols/2,rows/2), -smoothed_angle, 1)
        dst = cv2.warpAffine(img_steering,M,(cols,rows))
        cv2.imshow("Steering Wheel", dst)

        fig.canvas.draw_idle()
        plt.pause(0.1)
            
    # cv2.destroyAllWindows()


def get_activations(save_err, model_nvidia,img_lrg,rgb,model_dir,starting_frame,last_frame,baseline):

    i = starting_frame   # ----------   start from frame = 3000 ---------------
    smoothed_angle = 0

    act_list_1=[]; act_list_2=[]; act_list_3=[]; act_list_4=[]; act_list_5=[]; err_list=[]; steering_list=[];
    
    # graph = tf.Graph()

    # with graph.as_default():
        # model_nvidia = model_nvidia;
    tf.reset_default_graph()

    # from nets import model_nvidia
    from nets.pilotNet_5_layer_128_new import PilotNet
    # from nets.pilotNet_5_layer2 import PilotNet


    model_nvidia = PilotNet()
    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver() 

    # saver.build()
    # graph.finalize()
    # 
    # with tf.InteractiveSession() as sess:
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_dir)   

        while((cv2.waitKey(1) != ord('Q')) & (i < last_frame)):
            if (img_lrg == True):
                if (rgb == True):
                    full_image = scipy.misc.imread(dataset_dir + "/" + str(i) + ".jpg")
                    image = scipy.misc.imread(dataset_dir + "/" + str(i) + ".jpg", mode="RGB")[-128:] / 255.0;  #, flatten=True
                    image = scipy.misc.imresize(image,[102, 364])
                    camera_image = image;
                else:
                    full_image = scipy.misc.imread(dataset_dir + "/" + str(i) + ".jpg")
                    image = scipy.misc.imread(dataset_dir + "/" + str(i) + ".jpg", flatten=True)[-128:] / 255.0;  #, flatten=True
                    image = scipy.misc.imresize(image,[102, 364])
                    camera_image = image;
                    image = np.expand_dims(image, axis=-1);
            else:
                full_image = scipy.misc.imread(dataset_dir + "/" + str(i) + ".jpg",mode="RGB" ) # mode="RGB")
                image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
                camera_image = image


            conv1act, conv2act, conv3act, conv4act, conv5act , streeing_out = sess.run(
                fetches=[model_nvidia.h_conv1, model_nvidia.h_conv2, model_nvidia.h_conv3, model_nvidia.h_conv4, model_nvidia.h_conv5, model_nvidia.steering_output], 
                feed_dict={ 
                    model_nvidia.x: [image],
                    model_nvidia.keep_prob: 1.0
                }
            ) 
            
            # print ("-------------------", h_conv1)
            averageC5_n = tf.reduce_mean(conv5act, 3)
            # averageC5_n = tf.image.per_image_standardization(averageC5_n);
            averageC5_n = tf.expand_dims(averageC5_n, -1)

            averageC4_n = tf.reduce_mean(conv4act, 3)
            averageC4_n = tf.expand_dims(averageC4_n, -1)

            averageC3_n = tf.reduce_mean(conv3act, 3)
            averageC3_n = tf.expand_dims(averageC3_n, -1)
                
            averageC2_n = tf.reduce_mean(conv2act, 3)
            averageC2_n = tf.expand_dims(averageC2_n, -1)

            averageC1_n = tf.reduce_mean(conv1act, 3)
            averageC1_n = tf.image.per_image_standardization(averageC1_n);
            averageC1_n = tf.expand_dims(averageC1_n, -1)

            act_list_5.append(averageC5_n)
            act_list_4.append(averageC4_n)
            act_list_3.append(averageC3_n)
            act_list_2.append(averageC2_n)
            act_list_1.append(averageC1_n)


            degrees = streeing_out[0][0] * 180.0 / scipy.pi;
            smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
            # smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
            
            # call("clear")
            print("Predicted SWA: ", streeing_out , degrees) #, streeing_out1, streeing_out[0][0]- streeing_out1[0][0]    ) # + str(degrees))
            print("Smoothed SWA: " + str(smoothed_angle))
            print("Baseline SWA: " + baseline[i])
            print("SWA Error: " + str(degrees - float(baseline[i])))
            print("Frame #:", i)
            this_error = degrees - float(baseline[i])
            err_list.append(degrees)
            steering_list.append(float(baseline[i]))

            if (save_err == True):
                txt_file = open('../data/'+'/err_labels_3.txt', 'a')
                img_name = str(i) + ".jpg"
                txt_file.write(img_name + " " +str(this_error)+ "\n")
                txt_file.close() 
            
            if (save_err == True):
                txt_file = open('../data/'+'/pred_labels_3.txt', 'a')
                img_name = str(i) + ".jpg"
                txt_file.write(img_name + " " +str(degrees)+ "\n")
                txt_file.close() 

            i+=1

    sess.close()
    return [act_list_1, act_list_2,act_list_3,act_list_4,act_list_5, err_list,steering_list]
  
def get_saliency(save_saliency,act_list_1, act_list_2,act_list_3,act_list_4,act_list_5,model_nvidia,img_lrg,rgb,model_dir,starting_frame,total_f):
    
    i = 0;
    test_mask_out = [];
    saver = tf.train.Saver()
    while (i < total_f -2):

                
        averageC5_n = act_list_5[i]
        averageC4_n = act_list_4[i]
        averageC3_n = act_list_3[i]
        averageC2_n = act_list_2[i]
        averageC1_n = act_list_1[i]

        W_conv5_av = weight_variable_vis([3, 3,1,1])
        W_conv4_av = weight_variable_vis([3, 3,1,1])
        W_conv3_av = weight_variable_vis([5, 5,1,1])
        W_conv2_av = weight_variable_vis([5, 5,1,1])
        W_conv1_av = weight_variable_vis([5, 5,1,1])

        if (img_lrg == True):

            deconv5_avg = tf.nn.conv2d_transpose(value= averageC5_n, filter= W_conv5_av, output_shape=[1,8,40,1], strides=[1,1,1,1], padding='VALID');
            multC45 = tf.multiply(averageC4_n, deconv5_avg)  # use tensor multiplication.. not nm

            deconv4_avg = tf.nn.conv2d_transpose(value= multC45, filter= W_conv4_av, output_shape=[1,10,42,1], strides=[1,1,1,1], padding='VALID');
            multC34 = tf.multiply(averageC3_n, deconv4_avg)  # use tensor multiplication.. not nm

            deconv3_avg = tf.nn.conv2d_transpose(value= multC34, filter= W_conv3_av, output_shape=[1,23,88,1], strides=[1,2,2,1], padding='VALID');
            multC23 = tf.multiply(averageC2_n, deconv3_avg)  # use tensor multiplication.. not nm

            deconv2_avg = tf.nn.conv2d_transpose(value= multC23, filter= W_conv2_av, output_shape=[1,49,180,1], strides=[1,2,2,1], padding='VALID');
            multC12 = tf.multiply(averageC1_n, deconv2_avg)  # use tensor multiplication.. not nm

            the_mask = tf.nn.conv2d_transpose(value= multC12, filter= W_conv1_av, output_shape=[1,102,364,1], strides=[1,2,2,1], padding='VALID');
        else: 
            deconv5_avg = tf.nn.conv2d_transpose(value= averageC5_n, filter= W_conv5_av, output_shape=[1,3,20,1], strides=[1,1,1,1], padding='VALID');
            multC45 = tf.multiply(averageC4_n, deconv5_avg)  # use tensor multiplication.. not nm

            deconv4_avg = tf.nn.conv2d_transpose(value= multC45, filter= W_conv4_av, output_shape=[1,5,22,1], strides=[1,1,1,1], padding='VALID');
            multC34 = tf.multiply(averageC3_n, deconv4_avg)  # use tensor multiplication.. not nm

            deconv3_avg = tf.nn.conv2d_transpose(value= multC34, filter= W_conv3_av, output_shape=[1,14,47,1], strides=[1,2,2,1], padding='VALID');
            multC23 = tf.multiply(averageC2_n, deconv3_avg)  # use tensor multiplication.. not nm

            deconv2_avg = tf.nn.conv2d_transpose(value= multC23, filter= W_conv2_av, output_shape=[1,31,98,1], strides=[1,2,2,1], padding='VALID');
            multC12 = tf.multiply(averageC1_n, deconv2_avg)  # use tensor multiplication.. not nm

            the_mask = tf.nn.conv2d_transpose(value= multC12, filter= W_conv1_av, output_shape=[1,66,200,1], strides=[1,2,2,1], padding='VALID');


        init = tf.initialize_all_variables()
        
        # gg = tf.Graph()
        # with tf.InteractiveSession(gg.as_default) as sess:            
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            
            sess.run(init)  

            saver.restore(sess, model_dir)

            sess.run(W_conv5_av.initializer)
            sess.run(W_conv4_av.initializer)
            sess.run(W_conv3_av.initializer)
            sess.run(W_conv2_av.initializer)
            sess.run(W_conv1_av.initializer)

            test_mask = sess.run(the_mask)
            # test_mask = sess.run(fetch=[the_mask], 
            #                     feed={deconv.averageC1_n:averageC1_n,
            #                           deconv.averageC2_n:averageC2_n, 
            #                           deconv.averageC3_n:averageC3_n,
            #                           deconv.averageC4_n:averageC4_n,
            #                           deconv.averageC5_n:averageC5_n})

        sess.close();        
        test_mask_img = np.mean(test_mask, axis=3).squeeze(axis=0)
        salient_mask = (test_mask_img - np.min(test_mask_img))/(np.max(test_mask_img) - np.min(test_mask_img) + 1e-6)
        salient_mask2 = (test_mask_img)/(np.max(test_mask_img) - np.min(test_mask_img) + 1e-6)

        # im = Image.fromarray(salient_mask2 , 'F')
        # im.save("your_file"+str(i)+".jpg")
        if (save_saliency == True):
            scipy.misc.toimage(salient_mask2, cmin=0.0, cmax=1.0).save('../data/saliency_dataset_3/'+str(starting_frame+i)+'.jpg')

        test_mask_out.append([salient_mask2])
        print (starting_frame+i)
        i += 1;

   
    return test_mask_out


def get_saliency_dataset(network): 
    starting_frame_ = 0;
    last_frame_ = 0;

    for i in range(6000):
        start_time = time.time()

        starting_frame_ = last_frame_
        last_frame_ += 13;

        visual_back_prop2(network, starting_frame= starting_frame_, last_frame = last_frame_)    # 5_layer_66_rgb     5_layer_128_rgb

        print ("Totol: ", (time.time() - start_time)/60, " minutes")

    cv2.destroyAllWindows()


def run_both_models(network): 

    f = open(steer_baseline, 'r')
    baseline_raw = f.read().splitlines()
    baseline = [x.split(' ')[1] for x in baseline_raw]
    f.close()

    smoothed_angle = 0

    starting_frame_ = 0;
    last_frame_ = 11000;

    teacher_network = network;
    student_network = network;

    the_starting_frame = last_frame_;

    for i in range(1):
        start_time = time.time()

        starting_frame_ = last_frame_;
        last_frame_ += 13;

        visual_back_prop3(teacher_network, starting_frame_,last_frame_,baseline,smoothed_angle)  

        print ("Totol: ", (time.time() - start_time)/60, " minutes")

    cv2.destroyAllWindows()

    return 0

def show_result(): 

    data_path = '../data/datasets/driving_dataset/'
    saliency_dir = '../data/saliency_dataset_3/'
    sali =[]; imgs = []; angles = []; err_angl = [];

    with open(data_path + "data.txt") as f:
        for line in f:
            if (os.path.exists(saliency_dir + line.split()[0])):
                angl = float(line.split()[1]) * scipy.pi / 180
                imgs.append(data_path + line.split()[0])
                sali.append(saliency_dir + line.split()[0])
                angles.append(angl)

    with open(saliency_dir + "err_labels_3.txt") as f:
        for line in f:
            if (os.path.exists(saliency_dir + line.split()[0])):
                err = float(line.split()[1]) * scipy.pi / 180
                err_angl.append(err)

    smoothed_angle = angles[0]
    for i in range(20000,21000):
        degrees = angles[i] + err_angl[i];
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
        
        call("clear")
        print("Predicted SWA: ", str(degrees)) #, streeing_out1, streeing_out[0][0]- streeing_out1[0][0]    ) # + str(degrees))
        print("Smoothed SWA: " + str(smoothed_angle))
        print("Baseline SWA: " + str(angles[i]))
        
        image = scipy.misc.imread(imgs[i],mode="RGB" ) # mode="RGB")
        image = scipy.misc.imresize(image[-128:], [102, 364])

        salient_mask2 = scipy.misc.imread(sali[i],mode="RGB" ) # mode="RGB")
        salient_mask2 = scipy.misc.imresize(salient_mask2[-128:], [102, 364])

        if (i == 20000):
            fig = plt.figure()
            plt.subplot(211)
            plt.title('Camera Input and Saliency Map')
            camera_obj = plt.imshow(image)
            plt.subplot(212)
            saliency_obj = plt.imshow(salient_mask2) # , cmap='gray'

        camera_obj.set_data(image)
        saliency_obj.set_data(salient_mask2)

        # M = cv2.getRotationMatrix2D((cols/2,rows/2), -smoothed_angle, 1)
        # dst = cv2.warpAffine(img,M,(cols,rows))
        # cv2.imshow("Steering Wheel", dst)

        # im.set_data(image)
        fig.canvas.draw_idle()
        plt.pause(0.01)
        # draw()



# ------------- run the VisualBackProp  --------------
# visual_back_prop('5_layer_66_rgb', starting_frame= 20010)    #      5_layer_66_rgb   5_layer_128_rgb

# ------------- run the VisualBackProp and saves a Saliency Map dataset  --------------
# get_saliency_dataset('5_layer_128_gray')    #      5_layer_66_rgb   5_layer_128_rgb

# show_result()














