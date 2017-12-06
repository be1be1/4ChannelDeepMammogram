import numpy as np
from PIL import Image, ImageOps
from copy import copy


class ImageDataGenerator(object):

    def __init__(self, class_list, nb_classes, input_ch, horizontal_flip=False, shuffle=False,
                 histogram_eq=False, mean=np.array([55., 55., 55.]), scale_size=(227, 227)):

        # Init params
        self.histogram_eq = histogram_eq
        self.horizontal_flip = horizontal_flip
        self.nb_classes = nb_classes
        self.input_ch = input_ch
        self.num_ch = len(input_ch)
        self.shuffle = shuffle
        self.mean = mean
        self.scale_size = scale_size
        self.pointer = 0

        self.read_class_list(class_list)

        if self.shuffle:
            self.shuffle_data()

    def read_class_list(self, class_list):
        """
        Scan the image file and get the image paths and labels
        """
        with open(class_list) as f:
            lines = f.readlines()
            self.images = self.generate_empty_lst()
            self.labels = []

            for l in lines:
                items = l.split()
                for index, channel in enumerate(self.input_ch):
                    self.images[index].append(items[channel])

                self.labels.append(int(items[-1]))

            # store total number of data
            self.data_size = len(self.labels)


    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        images = self.generate_empty_lst()

        for index, ch_input in enumerate(self.images):
            images[index] = copy(ch_input)

        labels = copy(self.labels)

        self.images = self.generate_empty_lst()
        self.labels = []

        # create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(labels))

        for i in idx:
            for index in range(self.num_ch):
                self.images[index].append(images[index][i])

            self.labels.append(labels[i])


    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0

        if self.shuffle:
            self.shuffle_data()


    def generate_empty_lst(self):
        list = []
        for i in range(self.num_ch):
            list.append([])

        return list


    def next_batch(self, batch_size):
        """
        This function gets the next n ( = batch_size) images from the path list
        and labels and loads the images into them into memory
        """
        # Get next batch of image (path) and labels
        paths = self.generate_empty_lst()
        for i in range(self.num_ch):
            paths[i] = self.images[i][self.pointer:self.pointer + batch_size]

        labels = self.labels[self.pointer:self.pointer + batch_size]

        # update pointer
        self.pointer += batch_size

        # Read images
        images = np.ndarray([self.num_ch, batch_size, self.scale_size[0], self.scale_size[1], 3])


        for i in range(self.num_ch):
            # img = cv2.imread(paths[i])
            for j in range(len(paths[0])):

                img = Image.open(paths[i][j])

                if self.histogram_eq:
                    img = ImageOps.equalize(img)

                if self.horizontal_flip and np.random.random() < 0.5:
                    img = ImageOps.mirror(img)

                img = img.convert("RGB")
                img = img.resize((self.scale_size[0], self.scale_size[1]), resample=Image.LANCZOS)
                img = np.ndarray.astype(np.array(img), np.float32)
                img -= self.mean
                images[i][j] = img


        # Expand labels to one hot encoding
        one_hot_labels = np.zeros((batch_size, self.nb_classes))

        for i in range(len(labels)):
            one_hot_labels[i][labels[i]-1] = 1

        # return array of images and labels
        return images, one_hot_labels
