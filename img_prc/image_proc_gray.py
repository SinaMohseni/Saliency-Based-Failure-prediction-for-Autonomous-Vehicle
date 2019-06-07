import numpy as np
import scipy.misc
import random
from sklearn.model_selection import train_test_split
import cv2, os
import matplotlib.image as mpimg


class camera_image(object):
    """Preprocess images of the road ahead ans steering angles."""

    def __init__(self, tracks, laps, data_dir,test_size):
        # points to the end of the last batch, train & validation
        self.train_batch_pointer = 0
        self.val_batch_pointer = 0

        for the_track in tracks:
            for the_lap in laps:
                csv_file = data_dir + "/" + the_track + "/" + the_lap + \
                           "/driving_log.csv";
                data_df = pd.read_csv(csv_file)
                
                C_adrs = data_dir + "/" + the_track + "/" + the_lap + \
                           data_df['center'].values.split("/")[-1]
                L_adrs = data_dir + "/" + the_track + "/" + the_lap + \
                           data_df['left'].values.split("/")[-1]
                R_adrs = data_dir + "/" + the_track + "/" + the_lap + \
                           data_df['right'].values.split("/")[-1]

                X.append([C_adrs, L_adrs, R_adrs]);  # data_df[['center', 'left', 'right']].values
                y = data_df['steering'].values

        # shuffle and split
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, shuffle=True)  # random_state =0

        self.train_imgs, self.train_angles = dataset_synth(data_dir, X_train, y_train, is_training=True)
        self.val_imgs, self.val_angles = dataset_synth(data_dir, X_valid, y_valid, is_training=True)

        # get number of images
        self.num_images = len(X) * 3;
        print ("--------------- Training Data Size: ", self.num_images , "  -------------------------")

        self.num_train_images = len(self.train_imgs);
        self.num_val_images = len(self.val_imgs);

        self.im_size = scipy.misc.imread('../data/datasets/driving_dataset'+"/0.jpg").shape   # getting the image size
        print ("--------------- Training Image Size: ",self.im_size)
    

    def dataset_synth(data_dir, image_paths, steering_angles, batch_size, is_training):
        """
        Generate training image give image paths and associated steering angles
        """
        images = []; # np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
        steers = []; # np.empty(batch_size)
        # shuffle and synthesize the training/validation
        for index in np.random.permutation(image_paths.shape[0]):   # all csv lines
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # data_synth = 0.7*agm_L/R + 0.7*agm_cntr + 0.7*org_cntr ~= 2/3 total (L+R+C)
            rand_synth = np.random.rand(3);
            if is_training:
                if rand_synth[0] < 0.7:
                    image, steering_angle = augument_LR(data_dir, left, right, steering_angle); # rnd(pick,flip,brightness)
                    images.append(image)
                    steers.append(steering_angle)
                if rand_synth[1] < 0.7:
                    image, steering_angle = augument_C(data_dir, center,steering_angle); # flip + rnd(brightness)
                    images.append(image)
                    steers.append(steering_angle)
                if rand_synth[2] < 0.7:
                    image = load_image(data_dir, center);
                    images.append(image)
                    steers.append(steering_angle)
            else:
                image = load_image(data_dir, center)
        return images, steers

    def load_image(data_dir, image_file):
        return mpimg.imread(os.path.join(data_dir, image_file.strip()))

    def pick_one(data_dir, left, right, steering_angle):
        
        # randomly pick one of L/R cameras for augmentation

        choice = np.random.choice(2)
        if choice == 0:
            return load_image(data_dir, left), steering_angle + 0.2
        return load_image(data_dir, right), steering_angle - 0.2


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


    def augument_LR(data_dir, left, right, steering_angle):
        """
        generate randomly augumented images and adjust steering angle accordingly.
        """
        image, steering_angle = pick_one(data_dir, center, left, right, steering_angle)
        image, steering_angle = random_flip(image, steering_angle)
        image = random_brightness(image)
        return image, steering_angle

    def augument_C(data_dir, center, steering_angle):
        """
        generate randomly augumented images and adjust steering angle accordingly.
        """
        image = load_image(data_dir, center)
        steering_angle = steering_angle;

        image, steering_angle = flip(image, steering_angle)
        image = random_brightness(image)
        return image, steering_angle


    
    def load_train_batch(self, batch_size):
        batch_imgs = []
        batch_angles = []

        for i in range(0, batch_size):
            batch_imgs.append([self.train_imgs[(self.train_batch_pointer + i) % self.num_train_images]])
            batch_angles.append([self.train_angles[(self.train_batch_pointer + i) % self.num_train_images]])
        self.train_batch_pointer += batch_size
        return batch_imgs, batch_angles

    def load_val_batch(self, batch_size):
        batch_imgs = []
        batch_angles = []

        for i in range(0, batch_size):

            batch_imgs.append([self.val_imgs[(self.val_batch_pointer + i) % self.num_val_images]])
            batch_angles.append([self.val_angles[(self.val_batch_pointer + i) % self.num_val_images]])
        self.val_batch_pointer += batch_size
        return batch_imgs, batch_angles
