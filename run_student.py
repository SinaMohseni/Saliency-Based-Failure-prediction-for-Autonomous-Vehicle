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



def run_student(network,starting_frame):
    
    if network == '5_layer_128_rgb':
        from nets import model_nvidia_128_rgb as model_nvidia
        model_dir = './logs/checkpoint_5_layer_lrg4/checkpoint/model.ckpt'   # checkpoint_5_layer_lrg2-sina-RGB
        img_lrg = True
    elif network == '5_layer_128_gray':
        from nets.pilotNet_5_student_gray import PilotNet
        model_dir = './logs/student_7/checkpoint/model_2.ckpt'
        img_lrg = True
        saliency_dir =  '../data/saliency_dataset_3'
        dataset_dir = '../data/datasets/driving_dataset'

    elif network == '5_layer_128_gray_camera':
        from nets.pilotNet_5_student_gray import PilotNet
        model_dir = './logs/student_5/checkpoint/model.ckpt'
        img_lrg = True
        saliency_dir =  '../data/datasets/driving_dataset'  # '../data/saliency_dataset_3'
        dataset_dir = '../data/datasets/driving_dataset'

    elif network == '5_layer_66_rgb':  
        from nets.pilotNet_5_student_rgb import PilotNet   
        model_dir = './logs/student_3/checkpoint/model_2.ckpt'   # 
        img_lrg = False
        rgb = True
        saliency_dir = '../data/saliency_dataset2'
    elif network == '5_layer_66_gray':  # 5_layer_66_rgb old trained model
        from nets import model_nvidia_5_layer_66_gray as model_nvidia 
        model_dir = './logs/student_1/model_2.ckpt'  
        img_lrg = False
    


    f = open(error_baseline, 'r')
    baseline_raw = f.read().splitlines()
    baseline_err = [x.split(' ')[1] for x in baseline_raw]
    baseline_err_2 = [x.split(' ')[0] for x in baseline_raw]
    f.close()

    f = open(steer_baseline, 'r')
    baseline_raw = f.read().splitlines()
    baseline = [x.split(' ')[1] for x in baseline_raw]
    baseline_2 = [x.split(' ')[0] for x in baseline_raw]
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
    smoothed_pred_stu = 0.001 
    smoothed_base_err = 0.001

    i = starting_frame
    student_err_avg_1 = [];     student_err_avg_2 = [];     student_err_avg_3 = []; student_err_avg_4 = [];     student_err_avg_5 = [];     student_err_avg_6 = []; student_err_avg_7 = [];     student_err_avg_8 = [];     student_err_avg_9 = [];  student_err_avg_10 = []
    teacher_err_avg_1 = [];     teacher_err_avg_2 = [];     teacher_err_avg_3 = [];teacher_err_avg_4 = [];     teacher_err_avg_5 = [];     teacher_err_avg_6 = [];teacher_err_avg_7 = [];     teacher_err_avg_8 = [];     teacher_err_avg_9 = [];teacher_err_avg_10 = [];

    TP_1_5 = 0; FP_1_5 = 0; TN_1_5 = 0; FN_1_5 = 0;
    TP_1_7 = 0; FP_1_7 = 0; TN_1_7 = 0; FN_1_7 = 0;
    TP_1_10 = 0; FP_1_10 = 0; TN_1_10 = 0; FN_1_10 = 0;

    TP_2_5 = 0; FP_2_5 = 0; TN_2_5 = 0; FN_2_5 = 0;
    TP_2_7 = 0; FP_2_7 = 0; TN_2_7 = 0; FN_2_7 = 0;
    TP_2_10 = 0; FP_2_10 = 0; TN_2_10 = 0; FN_2_10 = 0;

    TP_3_5 = 0; FP_3_5 = 0; TN_3_5 = 0; FN_3_5 = 0;
    TP_3_7 = 0; FP_3_7 = 0; TN_3_7 = 0; FN_3_7 = 0;
    TP_3_10 = 0; FP_3_10 = 0; TN_3_10 = 0; FN_3_10 = 0;


    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        
        saver.restore(sess, model_dir)
        print("Load session successfully")

      
        while(i < len(baseline_err)): 
                
            
            if (img_lrg == True):
                if (network == '5_layer_128_rgb'):
                    full_image = scipy.misc.imread(saliency_dir + "/" + str(i) + ".jpg",mode="RGB") # 
                    image = scipy.misc.imresize(full_image[-150:], [102, 364]) / 255.0

                elif (network == '5_layer_128_gray'):
                    full_image = scipy.misc.imread(saliency_dir + "/" + str(i) + ".jpg",flatten=True ) /255.0;
                    image = np.expand_dims(full_image, axis=-1);
                elif (network == '5_layer_128_gray_camera'):
                    full_image = scipy.misc.imread(saliency_dir + "/" + str(i) + ".jpg",flatten=True )[-128:]  /255.0;
                    full_image = scipy.misc.imresize(full_image,[102, 364])
                    image = np.expand_dims(full_image, axis=-1);
            else:
                if (rgb == True):
                    full_image = scipy.misc.imread(saliency_dir + "/" + str(i) + ".jpg")
                    full_image = scipy.misc.imresize(full_image, [66, 200])
                    full_image = np.stack((full_image,)*3, axis = -1)
                    image = full_image;
                else: 
                    full_image = scipy.misc.imread(dataset_dir + "/" + str(i) + ".jpg",flatten=True) # mode="RGB")
                    image = scipy.misc.imresize(full_image, [66, 200]) / 255.0
                    image = np.expand_dims(image, axis=-1); 

            camera_image = scipy.misc.imread(dataset_dir + "/" + str(i) + ".jpg", mode="RGB" ) # mode="RGB")

            
            steering_out = sess.run(
                fetches=[model.steering],
                feed_dict={
                    model.image_input: [image],
                    model.keep_prob: 1.0
                }
            )

            error_estimate = float(steering_out[0][0] * 180.0 )/ scipy.pi
            

            smoothed_pred_stu = error_estimate; # 0.1*error_estimate + 0.9*smoothed_pred_stu
            smoothed_base_err = float(baseline_err[i]); # 0.1*float(baseline_err[i]) + 0.9*smoothed_base_err

            if (baseline_2[i] == baseline_err_2[i]):
                
                student_err = error_estimate - float(baseline_err[i])
                smooth_student_err = student_err # smoothed_base_err - smoothed_pred_stu

                # --------------------- Evaluation -------------------------
                #    - because our model does not include route planning etc.
                #    - It is not fair to evaluate at sharp edges etc.

                # 1- Histogram of teacher performance on diff. road conditions 
                if (abs(float(baseline[i])) < 10):
                    teacher_err_avg_1.append(float(baseline_err[i])); 
                elif (abs(float(baseline[i])) < 20):
                    teacher_err_avg_2.append(float(baseline_err[i])); 
                elif (abs(float(baseline[i])) < 30):
                    teacher_err_avg_3.append(float(baseline_err[i])); 
                elif (abs(float(baseline[i])) < 40):
                    teacher_err_avg_4.append(float(baseline_err[i])); 
                elif (abs(float(baseline[i])) < 50):
                    teacher_err_avg_5.append(float(baseline_err[i])); 
                elif (abs(float(baseline[i])) < 60):
                    teacher_err_avg_6.append(float(baseline_err[i])); 
                elif (abs(float(baseline[i])) < 70):
                    teacher_err_avg_7.append(float(baseline_err[i])); 
                elif (abs(float(baseline[i])) < 80):
                    teacher_err_avg_8.append(float(baseline_err[i])); 
                elif (abs(float(baseline[i])) < 90):
                    teacher_err_avg_9.append(float(baseline_err[i])); 
                else: 
                    teacher_err_avg_10.append(float(baseline_err[i])); 

                # 2- Histogram of student performance on diff. conditions 

                if (abs(float(baseline[i])) < 10):
                    student_err_avg_1.append(smooth_student_err); 
                elif (abs(float(baseline[i])) < 20):
                    student_err_avg_2.append(smooth_student_err); 
                elif (abs(float(baseline[i])) < 30):
                    student_err_avg_3.append(smooth_student_err); 
                elif (abs(float(baseline[i])) < 40):
                    student_err_avg_4.append(smooth_student_err); 
                elif (abs(float(baseline[i])) < 50):
                    student_err_avg_5.append(smooth_student_err); 
                elif (abs(float(baseline[i])) < 60):
                    student_err_avg_6.append(smooth_student_err); 
                elif (abs(float(baseline[i])) < 70):
                    student_err_avg_7.append(smooth_student_err); 
                elif (abs(float(baseline[i])) < 80):
                    student_err_avg_8.append(smooth_student_err); 
                elif (abs(float(baseline[i])) < 90):
                    student_err_avg_9.append(smooth_student_err); 
                else: 
                    student_err_avg_10.append(smooth_student_err); 
                
                # 3- Number of failures on >5 thesholds and safety gain
                if ((float(baseline[i]) < 30) & (float(baseline[i]) > -30)):
                    if ( (abs(smoothed_base_err) > 5) & (abs(smoothed_pred_stu)>5) ):
                        TP_1_5 +=1 
                    elif ( (abs(smoothed_base_err) > 5) & (abs(smoothed_pred_stu)<5) ):
                        FP_1_5 +=1 
                    elif ( (abs(smoothed_base_err) < 5) & (abs(smoothed_pred_stu)>5) ):
                        FN_1_5 +=1 
                    else:
                        TN_1_5 +=1 

                elif ((float(baseline[i]) < 60) & (float(baseline[i]) > -60)):
                    if ( (abs(smoothed_base_err) > 5) & (abs(smoothed_pred_stu)>5) ):
                        TP_2_5 +=1
                    elif ( (abs(smoothed_base_err) > 5) & (abs(smoothed_pred_stu)<5) ):
                        FP_2_5 +=1 
                    elif ( (abs(smoothed_base_err) < 5) & (abs(smoothed_pred_stu)>5) ):
                        FN_2_5 +=1 
                    else:
                        TN_2_5 +=1 

                else: 
                    if ( (abs(smoothed_base_err) > 5) & (abs(smoothed_pred_stu)>5) ):
                        TP_3_5 +=1 
                    elif ( (abs(smoothed_base_err) > 5) & (abs(smoothed_pred_stu)<5) ):
                        FP_3_5 +=1
                    elif ( (abs(smoothed_base_err) < 5) & (abs(smoothed_pred_stu)>5) ):
                        FN_3_5 +=1
                    else:
                        TN_3_5 +=1

                # 3- Number of failures on >7 thesholds and safety gain
                if ((float(baseline[i]) < 30) & (float(baseline[i]) > -30)):
                    if ( (abs(smoothed_base_err) > 7) & (abs(smoothed_pred_stu)>7) ):
                        TP_1_7 +=1 
                    elif ( (abs(smoothed_base_err) > 7) & (abs(smoothed_pred_stu)<7) ):
                        FP_1_7 +=1 
                    elif ( (abs(smoothed_base_err) < 7) & (abs(smoothed_pred_stu)>7) ):
                        FN_1_7 +=1 
                    else:
                        TN_1_7 +=1 

                elif ((float(baseline[i]) < 60) & (float(baseline[i]) > -60)):
                    if ( (abs(smoothed_base_err) > 7) & (abs(smoothed_pred_stu)>7) ):
                        TP_2_7 +=1 
                    elif ( (abs(smoothed_base_err) > 7) & (abs(smoothed_pred_stu)<7) ):
                        FP_2_7 +=1 
                    elif ( (abs(smoothed_base_err) < 7) & (abs(smoothed_pred_stu)>7) ):
                        FN_2_7 +=1 
                    else:
                        TN_2_7 +=1
                else: 
                    if ( (abs(smoothed_base_err) > 7) & (abs(smoothed_pred_stu)>7) ):
                        TP_3_7 +=1 
                    elif ( (abs(smoothed_base_err) > 7) & (abs(smoothed_pred_stu)<7) ):
                        FP_3_7 +=1  
                    elif ( (abs(smoothed_base_err) < 7) & (abs(smoothed_pred_stu)>7) ):
                        FN_3_7 +=1 
                    else:
                        TN_3_7 +=1 

                # 3- Number of failures on >10 thesholds and safety gain
                if ((float(baseline[i]) < 30) & (float(baseline[i]) > -30)):
                    if ( (abs(smoothed_base_err) > 10) & (abs(smoothed_pred_stu)>10) ):
                        TP_1_10 +=1  
                    elif ( (abs(smoothed_base_err) > 10) & (abs(smoothed_pred_stu)<10) ):
                        FP_1_10 +=1 
                    elif ( (abs(smoothed_base_err) < 10) & (abs(smoothed_pred_stu)>10) ):
                        FN_1_10 +=1  
                    else:
                        TN_1_10 +=1  
                elif ((float(baseline[i]) < 60) & (float(baseline[i]) > -60)):
                    if ( (abs(smoothed_base_err) > 10) & (abs(smoothed_pred_stu)>10) ):
                        TP_2_10 +=1 
                    elif ( (abs(smoothed_base_err) > 10) & (abs(smoothed_pred_stu)<10) ):
                        FP_2_10 +=1 
                    elif ( (abs(smoothed_base_err) < 10) & (abs(smoothed_pred_stu)>10) ):
                        FN_2_10 +=1 
                    else:
                        TN_2_10 +=1 
                else:                                                   
                    if ((abs(smoothed_base_err) > 10) & (abs(smoothed_pred_stu)>10) ):
                        TP_3_10 +=1 
                    elif ((abs(smoothed_base_err) > 10) & (abs(smoothed_pred_stu)<10) ):
                        FP_3_10 +=1 
                    elif ( (abs(smoothed_base_err) < 10) & (abs(smoothed_pred_stu)>10) ):
                        FN_3_10 +=1  
                    else:
                        TN_3_10 +=1 




                if (i %100 == 0):
                    call("clear")
                    if (len(student_err_avg_1)>0):
                        print("\n Avg Student Error 1: " , (sum(map(abs, student_err_avg_1)) / len(student_err_avg_1)) )
                    if (len(student_err_avg_2)>0):
                        print("Avg Student Error 2: " , (sum(map(abs,student_err_avg_2)) / len(student_err_avg_2)) )
                    if (len(student_err_avg_3)>0):
                        print("Avg Student Error 3: " , (sum(map(abs,student_err_avg_3)) / len(student_err_avg_3)) )
                    if (len(student_err_avg_4)>0):
                        print("Avg Student Error 4: " , (sum(map(abs,student_err_avg_4)) / len(student_err_avg_4)) )
                    if (len(student_err_avg_5)>0):
                        print("Avg Student Error 5: " , (sum(map(abs,student_err_avg_5)) / len(student_err_avg_5)) )
                    if (len(student_err_avg_6)>0):
                        print("Avg Student Error 6: " , (sum(map(abs,student_err_avg_6)) / len(student_err_avg_6)) )
                    if (len(student_err_avg_7)>0):
                        print("Avg Student Error 7: " , (sum(map(abs,student_err_avg_7)) / len(student_err_avg_7)) )
                    if (len(student_err_avg_8)>0):
                        print("Avg Student Error 8: " , (sum(map(abs,student_err_avg_8)) / len(student_err_avg_8)) )
                    if (len(student_err_avg_9)>0):
                        print("Avg Student Error 9: " , (sum(map(abs,student_err_avg_9)) / len(student_err_avg_9)) )
                    if (len(student_err_avg_10)>0):
                        print("Avg Student Error 10: " , (sum(map(abs,student_err_avg_10)) / len(student_err_avg_10)) )


                    if (len(teacher_err_avg_1)>0):
                        print("\n Avg Teacher Error 1: " , (sum(map(abs, teacher_err_avg_1)) / len(teacher_err_avg_1)) )
                    if (len(teacher_err_avg_2)>0):
                        print("Avg Teacher Error 2: " , (sum(map(abs, teacher_err_avg_2)) / len(teacher_err_avg_2)) )
                    if (len(teacher_err_avg_3)>0):
                        print("Avg Teacher Error 3: " , (sum(map(abs,teacher_err_avg_3)) / len(teacher_err_avg_3)) )
                    if (len(teacher_err_avg_4)>0):
                        print("Avg Teacher Error 4: " , (sum(map(abs,teacher_err_avg_4)) / len(teacher_err_avg_4)) )
                    if (len(teacher_err_avg_5)>0):
                        print("Avg Teacher Error 5: " , (sum(map(abs,teacher_err_avg_5)) / len(teacher_err_avg_5)) )
                    if (len(teacher_err_avg_6)>0):
                        print("Avg Teacher Error 6: " , (sum(map(abs,teacher_err_avg_6)) / len(teacher_err_avg_6)) )
                    if (len(teacher_err_avg_7)>0):
                        print("Avg Teacher Error 7: " , (sum(map(abs,teacher_err_avg_7)) / len(teacher_err_avg_7)) )
                    if (len(teacher_err_avg_8)>0):
                        print("Avg Teacher Error 8: " , (sum(map(abs,teacher_err_avg_8)) / len(teacher_err_avg_8)) )
                    if (len(teacher_err_avg_9)>0):
                        print("Avg Teacher Error 9: " , (sum(map(abs,teacher_err_avg_9)) / len(teacher_err_avg_9)) )
                    if (len(teacher_err_avg_10)>0):
                        print("Avg Teacher Error 10: " , (sum(map(abs,teacher_err_avg_10)) / len(teacher_err_avg_10)) )

                    # TP >>> safery_gained_avg_3_10 +=1 # .append(1); 
                    # FP >>> unsafe_avg_3_10 +=1 # .append(1); 
                    # FN >>> fls_alrm_avg_3_10 +=1 # .append(1); 
                    # TN >> TN_avg_3_10 +=1 # .append(1); 
                    
                    print ('\n Threshold == 5')
                    if (FP_1_5>0):
                        print("TP: ", TP_1_5, "FP: " , "FN: " , FN_1_5 ,FP_1_5, "Recall (TPR): " ,TP_1_5/ (TP_1_5+FP_1_5), "FPR: ", FP_1_5 / (FP_1_5 + TN_1_5))
                    if (FP_2_5>0):
                        print("TP: " , TP_2_5, "FP: ", "FN: " ,FN_2_5 , FP_2_5, "Recall(TPR): " ,TP_2_5/ (TP_1_5+FP_2_5), "FPR: ", FP_2_5 / (FP_2_5 + TN_2_5))
                    if (FP_3_10>0):
                        print("TP: " , TP_3_5, "FP: ", "FN: " ,FN_3_5 ,FP_3_5, "Recall(TPR): " ,TP_3_5/ (TP_3_5+FP_3_5), "FPR: " , FP_3_5 / (FP_3_5 + TN_3_5))
                    
                    print ('\n Threshold == 7')
                    if ((TP_1_7>0) & (FP_1_7>0)):
                        print("TP: ", TP_1_7, "FP: " ,FP_1_7, "FN: " , FN_1_7 , "Recall (TPR): " ,TP_1_7/ (TP_1_7+FP_1_7), "FPR: ", FP_1_7 / (FP_1_7 + TN_1_7))
                    if ((TP_2_7>0) & (FP_2_7>0)):
                        print("TP: " , TP_2_7, "FP: ", FP_2_7,"FN: " , FN_2_7 , "Recall (TPR): " ,TP_2_7/ (TP_2_7+FP_2_7), "FPR: ", FP_2_7 / (FP_2_7 + TN_2_7))
                    if ((TP_3_7>0) & (FP_3_7>0)):
                        print("TP: " , TP_3_7, "FP: " ,FP_3_7,  "FN: " , FN_3_7, "Recall (TPR): " ,TP_3_7/ (TP_3_7+FP_3_7), "FPR: ", FP_3_7 / (FP_3_7 + TN_3_7))
                    
                    print ('\n Threshold == 10')
                    if ((TP_1_10>0) & (FP_1_10>0)):
                        print("\n TP: ", TP_1_10, "FP: " ,FP_1_10, "FN: " , FN_1_10 , "Recall (TPR): " ,TP_1_10/ (TP_1_10+FP_1_10), "FPR: ", FP_1_10 / (FP_1_10 + TN_1_10))
                    if ( (TP_2_10>0) & (FP_2_10>0)):
                        print("TP: " , TP_2_10, "FP: " ,FP_2_10, "FN: " , FN_2_10 , "Recall (TPR): " ,TP_2_10/ (TP_2_10+FP_2_10), "FPR: ", FP_2_10 / (FP_2_10 + TN_2_10))
                    if ((TP_3_10>0) & (FP_3_10>0)):
                        print("TP: " , TP_3_10, "FP: " ,FP_3_10, "FN: " , FN_3_10 , "Recall (TPR): " ,TP_3_10/ (TP_3_10+FP_3_10), "FPR: ", FP_3_10 / (FP_3_10 + TN_3_10))
                    
                    print(len(baseline_err), len(baseline), "Frame: ", i)
            else: 
                print ("-------- baseline and error not equal: " , baseline_2[i] , "  ", baseline_err_2[i])
            i += 1
            while(os.path.exists(saliency_dir + "/" + str(i) + ".jpg") == False):
                i += 1
            


    cv2.destroyAllWindows()




run_student('5_layer_128_gray', starting_frame= 1)     # 5_layer_66_rgb   5_layer_128_rgb  5_layer_128_gray
