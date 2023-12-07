import math

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os.path
import json
import scipy.misc
import skimage.transform
from skimage.transform import resize

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.


class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size=[32, 32, 3], rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        #TODO: implement constructor

        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.pic_num = 100
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.IMGs_list = []
        self.label_file = open(self.label_path)
        self.LABELs_dict = json.load(self.label_file)
        self.label_file.close()
        self.LABELs_list_in_order = []
        for i in range(0, self.pic_num):
            self.LABELs_list_in_order.append(self.LABELs_dict[str(i)])
            self.file_name = self.file_path + str(i) + '.npy'
            self.IMGs_list.append(np.load(self.file_name))
        self.IMGs = np.array(self.IMGs_list)
        self.LABELs_in_order = np.array(self.LABELs_list_in_order)
        self.IMGs = resize(self.IMGs, (self.pic_num, self.image_size[0], self.image_size[1], self.image_size[2]))
        self.batch_start = 0
        self.batch_end = self.batch_start + self.batch_size
        self.batch_num = 1
        self.epoch_num = 0
        self.end_epoch = False

        if self.shuffle == True:
            self.IMGs, self.LABELs_in_order = self.shuffle_images_labels(self.IMGs, self.LABELs_in_order)

    def shuffle_images_labels(self, images, labels):
        IMG_LABEL = np.array(list(zip(images, labels)))
        np.random.shuffle(IMG_LABEL)
        shuffled_images, shuffled_labels = zip(*IMG_LABEL)
        shuffled_images = np.array(shuffled_images)
        shuffled_labels = np.array(shuffled_labels)
        return shuffled_images, shuffled_labels

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method

        batch_images = []
        batch_labels = []

        if self.batch_end == self.pic_num:
            for i in range(self.batch_start, self.batch_end):
                image_temp = self.IMGs[i][:][:][:]
                label_temp = self.LABELs_in_order[i]
                image_temp = self.augment(image_temp)
                batch_images.append(image_temp)
                batch_labels.append(label_temp)
            self.batch_start = 0
            self.batch_end = self.batch_start + self.batch_size
            self.batch_num += 1
            self.end_epoch = True
            if self.end_epoch == True and self.shuffle == True:
                self.IMGs, self.LABELs_in_order = self.shuffle_images_labels(self.IMGs, self.LABELs_in_order)
            return (np.array(batch_images), np.array(batch_labels))

        elif self.batch_end > self.pic_num:
            self.batch_end = self.pic_num
            for i in range(self.batch_start, self.batch_end):
                image_temp = self.IMGs[i][:][:][:]
                label_temp = self.LABELs_in_order[i]
                image_temp = self.augment(image_temp)
                batch_images.append(image_temp)
                batch_labels.append(label_temp)
            self.batch_end = self.batch_start + self.batch_size - self.pic_num
            self.batch_start = 0
            self.end_epoch = True

        if self.batch_end < self.pic_num:
            for i in range(self.batch_start, self.batch_end):
                image_temp = self.IMGs[i][:][:][:]
                label_temp = self.LABELs_in_order[i]
                image_temp = self.augment(image_temp)
                batch_images.append(image_temp)
                batch_labels.append(label_temp)
            self.batch_start = self.batch_end
            self.batch_end = self.batch_start + self.batch_size
            self.batch_num += 1
            if self.end_epoch == True and self.shuffle == True:
                self.IMGs, self.LABELs_in_order = self.shuffle_images_labels(self.IMGs, self.LABELs_in_order)
            if self.end_epoch == True:
                self.epoch_num += 1
                self.end_epoch = False

        return (np.array(batch_images), np.array(batch_labels))

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        if (self.mirroring == True) and (np.random.random(1) > 0.3):
            img = np.flip(img, 1)
        if (self.rotation == True) and (np.random.random(1) > 0.3):
            img = np.rot90(img, np.random.randint(1, 4, 1))
        # if self.image_size != [32, 32, 3]:
        #    img = resize(img, (self.image_size[0], self.image_size[1], self.image_size[2]))

        return img

    def current_epoch(self):
        return self.epoch_num

    def class_name(self, int_label):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        label = self.class_dict[int_label]
        return label

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        fig_columns = 4
        fig_rows = math.ceil(self.batch_size / 4)
        figure = plt.figure(num='Batch Number: ' + str(self.batch_num) + ', Epoch Number: ' + str(self.epoch_num))
        sub_plots = []
        (batch_images, batch_labels) = self.next()

        for i in range(self.batch_size):
            sub_plots.append(figure.add_subplot(fig_rows, fig_columns, i+1))
            sub_plots[i].set_title(self.class_name(batch_labels[i]))
            plt.imshow(batch_images[i][:][:][:])

        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)
        plt.show()
'''

x = 'D:/FAU ^^/Courses/Deep learning/ws2023/exercise0_material/src_to_implement/exercise_data/'
y = 'D:/FAU ^^/Courses/Deep learning/ws2023/exercise0_material/src_to_implement/Labels.json'

image_generator = ImageGenerator(x, y, 50, shuffle=True, mirroring=False, rotation=False)
while True:
    print('hello world')
    request = input()
    if request == 'next':
        #(batch_images, batch_labels) = image_generator.next()
        #print(len(batch_images), len(batch_labels))
        aaa = image_generator.next()
        print(len(aaa[0]), len(aaa[1]))
        print(image_generator.current_epoch())
        #print(image_generator.current_epoch())
        #print(images.shape)
        #print(images[1].shape)
        #print(images[0][:][:][0])
        plt.imshow(aaa[0][0])
        plt.show()
        #print(image_generator.class_name(batch_labels[0]))

    if request == 'show':
        image_generator.show()

'''
