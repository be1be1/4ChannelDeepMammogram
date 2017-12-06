import numpy as np
import tensorflow as tf
import os
import sys
from Simple_CNN import Simple_CNN
#from ImageDataGenerator import ImageDataGenerator
from datetime import datetime
from PIL import Image, ImageOps

###############################################################################
"""
Configuration settings
"""

class ImageGenerator(object):

    def __init__(self, class_list, nb_classes, input_ch, horizontal_flip=False, shuffle=False,
                 mean=np.array([104., 117., 124.])):

        # Init params
        self.horizontal_flip = horizontal_flip
        self.nb_classes = nb_classes
        self.input_ch = input_ch
        self.num_ch = len(input_ch)
        self.shuffle = shuffle
        self.mean = mean
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

                self.labels.append(int(items[4]))

            # store total number of data
            self.data_size = len(self.labels)


    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        images = self.generate_empty_lst()

        for index, ch_input in enumerate(self.images):
            images[index] = ch_input.copy()

        labels = self.labels.copy()

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
        images = np.ndarray([self.num_ch, batch_size, 256, 256, 1])


        for i in range(self.num_ch):
            # img = cv2.imread(paths[i])
            for j in range(len(paths[0])):
                img = Image.open(paths[i][j])

                if self.horizontal_flip and np.random.random() < 0.5:
                    img = ImageOps.mirror(img)

                #img = img.convert("RGB")
                #img = img.resize((self.scale_size[0], self.scale_size[1]), resample=Image.LANCZOS)
                img = np.ndarray.astype(np.expand_dims(np.array(img), axis=2), np.float32)
                #img -= self.mean
                images[i][j] = img


        # Expand labels to one hot encoding
        one_hot_labels = np.zeros((batch_size, self.nb_classes))

        for i in range(len(labels)):
            one_hot_labels[i][labels[i]-1] = 1

        # return array of images and labels
        return images, one_hot_labels











# Path to the textfiles for the trainings and validation set
train_file = 'train.txt'
val_file = 'val.txt'

# Learning params
learning_rate = 0.001
num_epochs = 15
batch_size = 70
Train_all = True

# Network params
Input_channel = [3]
Num_ch = len(Input_channel)
#dropout_rate = 1.0
num_classes = 4

if Train_all:
    train_layers = ['conv1','conv2','conv3','conv4','conv5','fc6','fc7']

elif Num_ch == 1:
    train_layers = ['fc6','fc7','fc8']

else:
    train_layers = ['conv5','fc6','fc7','fc8']


# How often we want to write the tf.summary data to disk
display_step = 1

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "C:/Users/Jiankun/Desktop/AlexNet/filewriter"
checkpoint_path = "C:/Users/Jiankun/Desktop/AlexNet/checkpoint"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [Num_ch, batch_size, 256, 256, 1])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

#tf.reset_default_graph()

# Initialize model
CNN = []

for i in range(Num_ch):
    CNN.append(Simple_CNN(x[i], num_classes, Num_ch, i, train_layers))


#AN_OUTPUT = tf.concat([AN1_output, AN2_output, AN3_output, AN4_output], axis=1)
#AN_OUTPUT = tf.concat(AN_output_seg, axis=1)


# Link variable to model output
score = CNN[0].fc7

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)


# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initalize the data generator seperately for the training and validation set
train_generator = ImageGenerator(train_file, num_classes, Input_channel, shuffle = True)
val_generator = ImageGenerator(val_file, num_classes, Input_channel, shuffle = False)

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)


# Start Tensorflow session
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    #writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    #for i in range(Num_ch):
     #   CNN[i].load_initial_weights(sess)


    print("{} Start training...".format(datetime.now()))
    #print("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))

    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        step = 1

        while step < train_batches_per_epoch:

            # Get a batch of images and labels
            batch_xs, batch_ys = train_generator.next_batch(batch_size)

            # And run the training op
            sess.run(train_op, feed_dict = {x: batch_xs,
                                            y: batch_ys,
                                            })

            # Generate summary with the current batch of data and write to file


            step += 1


        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_generator.next_batch(batch_size)
            acc = sess.run(accuracy, feed_dict = {x: batch_tx,
                                                  y: batch_ty,
                                                  })
            test_acc += acc
            test_count += 1

        test_acc /= test_count
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))

        # Reset the file pointer of the image data generator
        val_generator.reset_pointer()
        train_generator.reset_pointer()

