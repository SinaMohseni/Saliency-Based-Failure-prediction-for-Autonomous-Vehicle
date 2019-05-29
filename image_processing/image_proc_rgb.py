import os
import scipy.misc
import random

class camera_image(object):
    """Preprocess images of the road ahead ans steering angles."""

    def __init__(self, data_dir):
        imgs = []
        angles = []

        # points to the end of the last batch, train & validation
        self.train_batch_pointer = 0
        self.val_batch_pointer = 0

        # read data.txt
        data_path = data_dir + "/"
        with open(data_path + "data.txt") as f:
            for line in f:
                angl = float(line.split()[1]) * scipy.pi / 180

                if ((angl < 1.0) & (angl > -1.0)):
                    imgs.append(data_path + line.split()[0])
                    # the paper by Nvidia uses the inverse of the turning radius,
                    # but steering wheel angle is proportional to the inverse of turning radius
                    # so the steering wheel angle in radians is used as the output
                    angles.append(angl)

        # shuffle list of images
        c = list(zip(imgs, angles))
        random.shuffle(c)
        imgs, angles = zip(*c)

        # get number of images
        self.num_images = len(imgs)

        self.train_imgs = imgs[:int(self.num_images * 0.8)]
        self.train_angles = angles[:int(self.num_images * 0.8)]

        self.val_imgs = imgs[-int(self.num_images * 0.2):]
        self.val_angles = angles[-int(self.num_images * 0.2):]

        self.num_train_images = len(self.train_imgs)
        self.num_val_images = len(self.val_imgs)

        self.im_size = scipy.misc.imread('../data/datasets/driving_dataset'+"/0.jpg").shape   # getting the image size

    def load_train_batch(self, batch_size):
        batch_imgs = []
        batch_angles = []


        for i in range(0, batch_size):

            # batch_imgs.append(scipy.misc.imresize(
            #     scipy.misc.imread(self.train_imgs[(self.train_batch_pointer + i) % self.num_train_images])[-150:],
            #     [66, 200]) / 255.0)

            img_read = scipy.misc.imread(self.train_imgs[(self.train_batch_pointer + i) % self.num_train_images] , mode="RGB")[-128:] / 255.0
            batch_imgs.append(scipy.misc.imresize(img_read, [102, 364])) # / 255.0)  128, 455,

            batch_angles.append([self.train_angles[(self.train_batch_pointer + i) % self.num_train_images]])
        self.train_batch_pointer += batch_size
        return batch_imgs, batch_angles

    def load_val_batch(self, batch_size):
        batch_imgs = []
        batch_angles = []


        for i in range(0, batch_size):
            # batch_imgs.append(scipy.misc.imresize(scipy.misc.imread(self.val_imgs[(self.val_batch_pointer + i) % self.num_val_images])[-150:], [66, 200]) / 255.0)
            
            img_read = scipy.misc.imread(self.val_imgs[(self.val_batch_pointer + i) % self.num_val_images], mode="RGB")[-128:] / 255.0;
            # batch_imgs.append(scipy.misc.imread(self.val_imgs[(self.val_batch_pointer + i) % self.num_val_images], flatten=1)[-128:]) # / 255.0) -1.0*self.im_size[0]/2
            
            batch_imgs.append(scipy.misc.imresize(img_read, [102, 364])) # / 255.0)  128, 455,            
            batch_angles.append([self.val_angles[(self.val_batch_pointer + i) % self.num_val_images]])
        self.val_batch_pointer += batch_size
        return batch_imgs, batch_angles
