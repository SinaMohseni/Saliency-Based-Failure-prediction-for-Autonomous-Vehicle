import cv2, os
import numpy as np
import matplotlib.image as mpimg


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def load_image(data_dir, image_file):
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))

def pick_one(data_dir, left, right, steering_angle):
    # randomly pick one of L/R cameras for augmentation

    choice = np.random.choice(2)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    # elif choice == 1:
    #     return load_image(data_dir, right), steering_angle - 0.2
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


def dataset_synth(data_dir, image_paths, steering_angles, batch_size, is_training):
    """
    Generate training image give image paths and associated steering angles
    """
    images = []; # np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = []; # np.empty(batch_size)
    
    for i in range(0, data_set_len):
        for index in np.random.permutation(image_paths.shape[0]):
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

