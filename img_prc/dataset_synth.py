import numpy as np
import pandas as pd
import scipy.misc
import random

from sklearn.model_selection import train_test_split
import cv2, os
import matplotlib.image as mpimg


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
CROP_TOP = 5
CROP_BOTTOM = 0 


# def dataset_synth_old(camera_images, steering_angles, is_training):

#     tot_images = [];
#     tot_steers = [];
#     # shuffle and synthesize the training/validation
#     for index in range(len(camera_images)): # np.random.permutation(image_paths.shape[0]):
        
#         center, left, right = camera_images[index];
#         steering_angle = steering_angles[index]

#         # data_synth = 0.7*agm_L/R + 0.7*agm_cntr + 0.7*org_cntr ~= 2/3 total (L+R+C)
#         rand_synth = np.random.rand(3);
#         if is_training:
#             if rand_synth[0] < 0.7:
#                 image, steering_angle = augument_LR(left, right, steering_angle); # rnd(pick,flip,brightness)
#                 tot_images.append(image)
#                 tot_steers.append(steering_angle)
#             if rand_synth[1] < 0.7:
#                 image, steering_angle = augument_C(center,steering_angle); # flip + rnd(brightness)
#                 tot_images.append(image)
#                 tot_steers.append(steering_angle)
#             if rand_synth[2] < 0.7:
#                 tot_images.append(load_image(center))
#                 tot_steers.append(steering_angle)
#         else:
#             tot_images.append(load_image(center))
#             tot_steers.append(steering_angle)

#     return tot_images, tot_steers

def dataset_synth(camera_images, steering_angles, is_training):

    tot_images = [];
    tot_steers = [];
    # shuffle and synthesize the training/validation
    for index in range(len(camera_images)): # np.random.permutation(image_paths.shape[0]):
        
        center, left, right = camera_images[index];
        steering_angle = steering_angles[index]

        # data_synth = 0.7*agm_L/R + 0.7*agm_cntr + 0.7*org_cntr ~= 2/3 total (L+R+C)
        rand_synth = np.random.rand(3);
        if is_training:
            if rand_synth[0] < 0.7:
                image, steering_angle = augument_LR(left, right, steering_angle); # rnd(pick,flip,brightness)
                tot_images.append(image)
                tot_steers.append(steering_angle)
            if rand_synth[1] < 0.7:
                image, steering_angle = augument_C(center,steering_angle); # flip + rnd(brightness)
                tot_images.append(image)
                tot_steers.append(steering_angle)
            if rand_synth[2] < 0.7:
                # tot_images.append(load_image(center))
                tot_images.append(center)
                tot_steers.append(steering_angle)
        else:
            # tot_images.append(load_image(center))
            tot_images.append(center)
            tot_steers.append(steering_angle)

    return tot_images, tot_steers

def load_image(image_file):
     image = cv2.imread(image_file)[CROP_BOTTOM:-CROP_TOP, :, :] # mpimg.imread  # os.path.join(data_dir, image_file.strip()) 
     image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
     return image

def pick_one(left, right, steering_angle):
    
    # randomly pick one of L/R cameras for augmentation

    choice = np.random.choice(2)
    if choice == 0:
        # return load_image(left), steering_angle + 0.25 - np.random.rand()/10;
        return left, steering_angle + 0.20 - np.random.rand()/10;   
    # return load_image(right), steering_angle - 0.25 + np.random.rand()/10;
    return right, steering_angle - 0.20 + np.random.rand()/10;


def random_flip(image, steering_angle):
    # randomly flip half of the images for augmentation
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def flip(image, steering_angle):
    # flip images for augmentation
    image = cv2.flip(image, 1)
    steering_angle = -steering_angle
    return image, steering_angle

def random_brightness(image):
    # Randomly adjust image brightness.
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augument_LR(left, right, steering_angle):
    """
    generate randomly augumented images and adjust steering angle accordingly.
    """
    image, steering_angle = pick_one(left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image = random_brightness(image)
    return image, steering_angle

def augument_C(center, steering_angle):
    """
    generate randomly augumented images and adjust steering angle accordingly.
    """
    # image = load_image(center)
    image = center

    image, steering_angle = flip(image, steering_angle)
    image = random_brightness(image)
    return image, steering_angle


class training_set(object):

    def __init__(self, data_dir,test_size):
        # points to the end of the last batch, train & validation
        self.train_batch_pointer = 0
        self.val_batch_pointer = 0

        self.all_x = []
        self.all_y = []
        self.test_size = test_size

        for the_track in data_dir:
            if os.path.isdir(the_track):
                Laps = os.listdir(the_track);
            else:
                Laps = [the_track];

            Laps = ['Lap_1']
            for the_lap in Laps:
                print ("...Loading "+ the_track + " " + the_lap + " ...")
                csv_file = the_track + the_lap + "/driving_log.csv";
                data_df = pd.read_csv(csv_file)

                C_adrs = [the_track + the_lap + "/IMG/"+ xx.split("\\")[-1] for xx in data_df['center']]
                L_adrs = [the_track + the_lap + "/IMG/"+ xx.split("\\")[-1] for xx in data_df['left']]
                R_adrs = [the_track + the_lap + "/IMG/"+ xx.split("\\")[-1] for xx in data_df['right']]

                for i in range(len(C_adrs)):
                    # self.all_x.append([C_adrs[i], L_adrs[i], R_adrs[i]])
                    self.all_x.append([load_image(C_adrs[i]), load_image(L_adrs[i]), load_image(R_adrs[i])])
                
                self.all_y = self.all_y + data_df['steering'].tolist();

        # print ('{}{}'.format("...Total Training data... ", len(all_y)));



    def shuffle_all(self):

        # shuffle and split
        X_train, X_valid, y_train, y_valid = train_test_split(self.all_x, self.all_y, test_size=self.test_size, shuffle=True)  # random_state =0
        # print ("---- Data Augmentation...")

        self.train_imgs, self.train_angles = dataset_synth(X_train, y_train, is_training=True)
        # print ("---- training set loaded", len(self.train_angles) , len(self.train_imgs))

        self.val_imgs, self.val_angles = dataset_synth(X_valid, y_valid, is_training=False)
        # print ("---- validation set loaded", len(self.val_angles) , len(self.val_imgs))

        # get number of training items
        self.num_images = len(y_train);
        # print ("--------------- Training Data Size: ", self.num_images , "  -------------------------")

        self.num_train_images = len(self.train_imgs);
        self.num_val_images = len(self.val_imgs);

        # self.im_size = mpimg.imread(X_train[0][0]).shape; # X_train[0].shape; # scipy.misc.imread('../data/datasets/driving_dataset'+"/0.jpg").shape   # getting the image size
        # print ("--------------- Training Image Size: ",self.im_size)

    def load_train_batch(self, batch_size):
        batch_imgs = []
        batch_angles = []

        for i in range(0, batch_size):
            batch_imgs.append(self.train_imgs[(self.train_batch_pointer + i) % self.num_train_images])
            batch_angles.append(self.train_angles[(self.train_batch_pointer + i) % self.num_train_images])
        self.train_batch_pointer += batch_size
        return batch_imgs, batch_angles

    def load_val_batch(self, batch_size):
        batch_imgs = []
        batch_angles = []

        for i in range(0, batch_size):

            batch_imgs.append(self.val_imgs[(self.val_batch_pointer + i) % self.num_val_images])
            batch_angles.append(self.val_angles[(self.val_batch_pointer + i) % self.num_val_images])
        self.val_batch_pointer += batch_size
        return batch_imgs, batch_angles
