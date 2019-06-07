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




def evaluation(network,starting_frame):
    
    if network == '5_layer_128_rgb':
        from nets import model_nvidia_128_rgb as model_nvidia
        model_dir = './logs/checkpoint_5_layer_lrg4/checkpoint/model.ckpt'   
        img_lrg = True
    elif network == '5_layer_128_gray':
        from nets.pilotNet_5_student_gray import PilotNet
        model_dir = './logs/student_5/checkpoint/model_2.ckpt'
        img_lrg = True
        saliency_dir = '../data/datasets/driving_dataset' 
        dataset_dir = '../data/datasets/driving_dataset'

    elif network == '5_layer_66_rgb':  
        from nets.pilotNet_5_student_rgb import PilotNet   
        model_dir = './logs/student_3/checkpoint/model_2.ckpt'   # 
        img_lrg = False
        rgb = True
        saliency_dir = '../data/saliency_dataset2'
    elif network == '5_layer_66_gray':  
        from nets import model_nvidia_5_layer_66_gray as model_nvidia 
        model_dir = './logs/student_1/model_2.ckpt'  
        img_lrg = False
    


    f = open(error_baseline, 'r')
    baseline_raw = f.read().splitlines()
    baseline_err = [x.split(' ')[1] for x in baseline_raw]
    f.close()

    f = open(steer_baseline, 'r')
    baseline_raw = f.read().splitlines()
    baseline = [x.split(' ')[1] for x in baseline_raw]
    f.close()

    f = open('../data/student_pred.txt', 'r')  
    baseline_raw = f.read().splitlines()
    student_pred = [x.split(' ')[1] for x in baseline_raw]
    f.close()

    WIN_MARGIN_LEFT = 240;
    WIN_MARGIN_TOP = 240;
    WIN_MARGIN_BETWEEN = 180;
    WIN_WIDTH = 480;

    img = cv2.imread(steer_image, 0)
    rows,cols = img.shape

    # Visualization init
    cv2.namedWindow("Steering Wheel", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Steering Wheel", WIN_MARGIN_LEFT, WIN_MARGIN_TOP)
    cv2.namedWindow("Scenario", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Scenario", WIN_MARGIN_LEFT+cols+WIN_MARGIN_BETWEEN, WIN_MARGIN_TOP)

    g = np.zeros((3,3), 'uint8')
    b = np.zeros((3,3), 'uint8')
    r=np.empty([3,3])
    r.fill(256);
    green = np.dstack((g,r,b))

    g = np.zeros((3,3), 'uint8')
    b = np.zeros((3,3), 'uint8')
    r=np.empty([3,3])
    r.fill(256);
    red = np.dstack((g,b,r))

    smoothed_angle = 0;
    smoothed_base = 0;
    smoothed_pred_stu = 0 
    smoothed_base_err = 0

    i = starting_frame
    student_err_avg_1 = [];     student_err_avg_2 = [];     student_err_avg_3 = []
    teacher_err_avg_1 = [];     teacher_err_avg_2 = [];     teacher_err_avg_3 = []

    unsafe_avg_1_5 = 0;  safery_gained_avg_1_5 = 0; fls_alrm_avg_1_5 = 0;
    unsafe_avg_2_5 = 0;  safery_gained_avg_2_5 = 0; fls_alrm_avg_2_5 = 0;
    unsafe_avg_3_5 = 0;  safery_gained_avg_3_5 = 0; fls_alrm_avg_3_5 = 0;

    unsafe_avg_1_7 = 0;  safery_gained_avg_1_7 = 0; fls_alrm_avg_1_7 = 0;
    unsafe_avg_2_7 = 0;  safery_gained_avg_2_7 = 0; fls_alrm_avg_2_7 = 0;
    unsafe_avg_3_7 = 0;  safery_gained_avg_3_7 = 0; fls_alrm_avg_3_7 = 0;

    unsafe_avg_1_10 = 0;  safery_gained_avg_1_10 = 0; fls_alrm_avg_1_10 = 0;
    unsafe_avg_2_10= 0;  safery_gained_avg_2_10 = 0; fls_alrm_avg_2_10 = 0;
    unsafe_avg_3_10 = 0;  safery_gained_avg_3_10 = 0; fls_alrm_avg_3_10 = 0;

    while(i < len(baseline_err)):
        
        error_estimate = float(student_pred[i]) 


        smoothed_pred_stu += 0.2 * pow(abs((error_estimate - smoothed_pred_stu)), 2.0 / 3.0) * (error_estimate - smoothed_pred_stu) / abs(error_estimate - smoothed_pred_stu)
        smoothed_base_err += 0.2 * pow(abs(( float(baseline_err[i]) - smoothed_base_err)), 2.0 / 3.0) * ( float(baseline_err[i]) - smoothed_base_err) / abs( float(baseline_err[i]) - smoothed_base_err)

        student_err = error_estimate - float(baseline_err[i])


        # --------------------- Evaluation -------------------------
        #    - because our model does not include route planning etc.
        #    - It is not fair to evaluate at sharp edges etc.

        # 1- Histogram of teacher performance on diff. road conditions 
        if ((float(baseline[i]) < 30) & (float(baseline[i]) > -30)):
            teacher_err_avg_1.append(float(baseline_err[i])); 
        elif ((float(baseline[i]) < 60) & (float(baseline[i]) > -60)):
            teacher_err_avg_2.append(float(baseline_err[i])); 
        else: 
            teacher_err_avg_3.append(float(baseline_err[i])); 

        # 2- Histogram of student performance on diff. conditions 
        if ((float(baseline[i]) < 30) & (float(baseline[i]) > -30)):
            student_err_avg_1.append(student_err); 
        elif ((float(baseline[i]) < 60) & (float(baseline[i]) > -60)):
            student_err_avg_2.append(student_err); 
        else: 
            student_err_avg_3.append(student_err); 
        
        # 3- Number of failures on >5 thesholds and safety gain
        if ((float(baseline[i]) < 30) & (float(baseline[i]) > -30)):
            if ( (abs(smoothed_base_err) > 5) & (abs(smoothed_pred_stu)>5) ):
                safery_gained_avg_1_5 +=1 # .append(1); 
            if ( (abs(smoothed_base_err) > 5) & (abs(smoothed_pred_stu)<5) ):
                unsafe_avg_1_5 +=1 # .append(1); 
            if ( (abs(smoothed_base_err) < 5) & (abs(smoothed_pred_stu)>5) ):
                fls_alrm_avg_1_5 +=1 # .append(1); 
        elif ((float(baseline[i]) < 60) & (float(baseline[i]) > -60)):
            if ( (abs(smoothed_base_err) > 5) & (abs(smoothed_pred_stu)>5) ):
                safery_gained_avg_2_5 +=1 # .append(1); 
            if ( (abs(smoothed_base_err) > 5) & (abs(smoothed_pred_stu)<5) ):
                unsafe_avg_2_5 +=1 # .append(1); 
            if ( (abs(smoothed_base_err) < 5) & (abs(smoothed_pred_stu)>5) ):
                fls_alrm_avg_2_5 +=1 # .append(1); 
        else: 
            if ( (abs(smoothed_base_err) > 5) & (abs(smoothed_pred_stu)>5) ):
                safery_gained_avg_3_5 +=1 # .append(1); 
            if ( (abs(smoothed_base_err) > 5) & (abs(smoothed_pred_stu)<5) ):
                unsafe_avg_3_5 +=1 # .append(1); 
            if ( (abs(smoothed_base_err) < 5) & (abs(smoothed_pred_stu)>5) ):
                fls_alrm_avg_3_5 +=1 # .append(1); 

        # 3- Number of failures on >7 thesholds and safety gain
        if ((float(baseline[i]) < 30) & (float(baseline[i]) > -30)):
            if ( (abs(smoothed_base_err) > 7) & (abs(smoothed_pred_stu)>7) ):
                safery_gained_avg_1_7 +=1 # .append(1); 
            if ( (abs(smoothed_base_err) > 7) & (abs(smoothed_pred_stu)<7) ):
                unsafe_avg_1_7 +=1 # .append(1); 
            if ( (abs(smoothed_base_err) < 7) & (abs(smoothed_pred_stu)>7) ):
                fls_alrm_avg_1_7 +=1 # .append(1); 
        elif ((float(baseline[i]) < 60) & (float(baseline[i]) > -60)):
            if ( (abs(smoothed_base_err) > 7) & (abs(smoothed_pred_stu)>7) ):
                safery_gained_avg_2_7 +=1 # .append(1); 
            if ( (abs(smoothed_base_err) > 7) & (abs(smoothed_pred_stu)<7) ):
                unsafe_avg_2_7 +=1 # .append(1); 
            if ( (abs(smoothed_base_err) < 7) & (abs(smoothed_pred_stu)>7) ):
                fls_alrm_avg_2_7 +=1 # .append(1); 
        else: 
            if ( (abs(smoothed_base_err) > 7) & (abs(smoothed_pred_stu)>7) ):
                safery_gained_avg_3_7 +=1 # .append(1); 
            if ( (abs(smoothed_base_err) > 7) & (abs(smoothed_pred_stu)<7) ):
                unsafe_avg_3_7 +=1 # .append(1); 
            if ( (abs(smoothed_base_err) < 7) & (abs(smoothed_pred_stu)>7) ):
                fls_alrm_avg_3_7 +=1 # .append(1); 

        # 3- Number of failures on >10 thesholds and safety gain
        if ((float(baseline[i]) < 30) & (float(baseline[i]) > -30)):
            if ( (abs(smoothed_base_err) > 10) & (abs(smoothed_pred_stu)>10) ):
                safery_gained_avg_1_10 +=1 # .append(1); 
            if ( (abs(smoothed_base_err) > 10) & (abs(smoothed_pred_stu)<10) ):
                unsafe_avg_1_10 +=1 # .append(1); 
            if ( (abs(smoothed_base_err) < 10) & (abs(smoothed_pred_stu)>10) ):
                fls_alrm_avg_1_10 +=1 # .append(1); 
        elif ((float(baseline[i]) < 60) & (float(baseline[i]) > -60)):
            if ( (abs(smoothed_base_err) > 10) & (abs(smoothed_pred_stu)>10) ):
                safery_gained_avg_2_10 +=1 # .append(1); 
            if ( (abs(smoothed_base_err) > 10) & (abs(smoothed_pred_stu)<10) ):
                unsafe_avg_2_10 +=1 # .append(1); 
            if ( (abs(smoothed_base_err) < 10) & (abs(smoothed_pred_stu)>10) ):
                fls_alrm_avg_2_10 +=1 # .append(1); 
        else: 
            if ( (abs(smoothed_base_err) > 10) & (abs(smoothed_pred_stu)>10) ):
                safery_gained_avg_3_10 +=1 # .append(1); 
            if ( (abs(smoothed_base_err) > 10) & (abs(smoothed_pred_stu)<10) ):
                unsafe_avg_3_10 +=1 # .append(1); 
            if ( (abs(smoothed_base_err) < 10) & (abs(smoothed_pred_stu)>10) ):
                fls_alrm_avg_3_10 +=1 # .append(1); 

        # # 3- Number of failures on >5 and >7 and >10 thesholds and safety gain
        if (i %100 == 0):
            call("clear")
            if (len(student_err_avg_1)>0):
                print("\n Avg Student Error 1: " , (sum(map(abs, student_err_avg_1)) / len(student_err_avg_1)) )
            if (len(student_err_avg_2)>0):
                print("Avg Student Error 2: " , (sum(map(abs,student_err_avg_2)) / len(student_err_avg_2)) )
            if (len(student_err_avg_3)>0):
                print("Avg Student Error 3: " , (sum(map(abs,student_err_avg_3)) / len(student_err_avg_3)) )

            if (len(teacher_err_avg_1)>0):
                print("\n Avg Teacher Error 1: " , (sum(map(abs, teacher_err_avg_1)) / len(teacher_err_avg_1)) )
            if (len(teacher_err_avg_2)>0):
                print("Avg Teacher Error 2: " , (sum(map(abs, teacher_err_avg_2)) / len(teacher_err_avg_2)) )
            if (len(teacher_err_avg_3)>0):
                print("Avg Teacher Error 3: " , (sum(map(abs,teacher_err_avg_3)) / len(teacher_err_avg_3)) )

            if (safery_gained_avg_1_5>0):
                print("\n Safe: ", safery_gained_avg_1_5, "Unsafe: " ,unsafe_avg_1_5, "Gained: " ,safery_gained_avg_1_5/ (safery_gained_avg_1_5+unsafe_avg_1_5), "false_alarm: " ,fls_alrm_avg_1_5)
            if (safery_gained_avg_2_5>0):
                print("Safe: " , safery_gained_avg_2_5, "Unsafe: " ,unsafe_avg_2_5, "Gained: " ,safery_gained_avg_2_5/ (safery_gained_avg_2_5+unsafe_avg_2_5), "false_alarm: " ,fls_alrm_avg_2_5)
            if (safery_gained_avg_3_5>0):
                print("Safe: " , safery_gained_avg_3_5, "Unsafe: " ,unsafe_avg_3_5, "Gained: " ,safery_gained_avg_3_5/ (safery_gained_avg_3_5+unsafe_avg_3_5), "false_alarm: " ,fls_alrm_avg_3_5)
            
            if (safery_gained_avg_1_7>0):
                print("\n Safe: ", safery_gained_avg_1_7, "Unsafe: " ,unsafe_avg_1_7, "Gained: " ,safery_gained_avg_1_7/ (safery_gained_avg_1_7+unsafe_avg_1_7), "false_alarm: " ,fls_alrm_avg_1_7)
            if (safery_gained_avg_2_7>0):
                print("Safe: " , safery_gained_avg_2_7, "Unsafe: " ,unsafe_avg_2_7, "Gained: " ,safery_gained_avg_2_7/ (safery_gained_avg_2_7+unsafe_avg_2_7), "false_alarm: " ,fls_alrm_avg_2_7)
            if (safery_gained_avg_3_7>0):
                print("Safe: " , safery_gained_avg_3_7, "Unsafe: " ,unsafe_avg_3_7, "Gained: " ,safery_gained_avg_3_7/ (safery_gained_avg_3_7+unsafe_avg_3_7), "false_alarm: " ,fls_alrm_avg_3_7)
            

            if (safery_gained_avg_1_10>0):
                print("\n Safe: ", safery_gained_avg_1_10, "Unsafe: " ,unsafe_avg_1_10, "Gained: " ,safery_gained_avg_1_10/ (safery_gained_avg_1_10+unsafe_avg_1_10), "false_alarm: " ,fls_alrm_avg_1_10)
            if (safery_gained_avg_2_10>0):
                print("Safe: " , safery_gained_avg_2_10, "Unsafe: " ,unsafe_avg_2_10, "Gained: " ,safery_gained_avg_2_10/ (safery_gained_avg_2_10+unsafe_avg_2_10), "false_alarm: " ,fls_alrm_avg_2_10)
            if (safery_gained_avg_3_10>0):
                print("Safe: " , safery_gained_avg_3_10, "Unsafe: " ,unsafe_avg_3_10, "Gained: " ,safery_gained_avg_3_10/ (safery_gained_avg_3_10+unsafe_avg_3_10), "false_alarm: " ,fls_alrm_avg_3_10)
            

            print("Frame: ", i)
        
        i += 1
        while(os.path.exists(saliency_dir + "/" + str(i) + ".jpg") == False):
            i += 1
        


    cv2.destroyAllWindows()




evaluation('5_layer_128_gray', starting_frame= 1)     # 5_layer_66_rgb   5_layer_128_rgb  5_layer_128_gray
