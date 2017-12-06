import numpy as np
import tensorflow as tf
import os
import sys
from Splitted_AlexNet import Splitted_AlexNet
from Merged_FC import Merged_FC
from ImageDataGenerator import ImageDataGenerator
from datetime import datetime
import matplotlib.pyplot as plt

"""
Configuration settings
"""
with tf.device("/gpu:0"):
# Path to the textfiles for the trainings and validation set
    train_file = 'train_1_ch.txt'
    val_file = 'val_1_ch.txt'
	#train_file = 'train_4_ch.txt'
	#val_file = 'val_4_ch.txt'

	# Learning params
    learning_rate = 1e-5
    batch_size = 64
    Train_all = False

	# Network params
	# 1 channel: [0], 2 channels: [0,1], 4 channels: [0,1,2,3]
    Input_channel = [0]
    Num_ch = len(Input_channel)
    dropout_rate = 0.5
    num_classes = 4
    num_epochs = 100
    w_decay = 1e-3

    if Train_all:
	    train_layers = ['conv1','conv2','conv3','conv4','conv5','fc6','fc7','fc8']
    elif Num_ch == 1:
	    train_layers = ['fc7','fc8']
    else:
	    train_layers = ['fc6','fc7','fc8']

	# How often we want to write the tf.summary data to disk
    display_step = 1

	# Path for tf.summary.FileWriter and to store model checkpoints
    filewriter_path = "/guest/z/zhan4554/Desktop/Alexnet/filewriter"
    checkpoint_path = "/guest/z/zhan4554/Desktop/Alexnet/checkpoint"

	# Create parent path if it doesn't exist
    if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

	# TF placeholder for graph input and output
    x = tf.placeholder(tf.float32, [Num_ch, batch_size, 227, 227, 3])
    y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    values = [0.001, 0.0001, 0.00001]

    #learning_rate = tf.train.exponential_decay(lr, global_step, 2000, 0.1, staircase=True)
    #learning_rate = tf.train.piecewise_constant(global_step, [10000, 20000], values)
	#tf.reset_default_graph()

	# Initialize model
    AlexNet = []
    AN_output_seg = []
    for i in range(Num_ch):
        AlexNet.append(Splitted_AlexNet(x[i], keep_prob, num_classes, Num_ch, i, train_layers))
        AN_output_seg.append(AlexNet[i].fc6)

	#AN_OUTPUT = tf.concat([AN1_output, AN2_output, AN3_output, AN4_output], axis=1)
    AN_OUTPUT = tf.concat(AN_output_seg, axis=1)


	# Link variable to model output
    score = Merged_FC(AN_OUTPUT, keep_prob, num_classes).fc8

	# List of trainable variables of the layers we want to train
    var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

    regularizer = tf.contrib.layers.l2_regularizer(w_decay)
    penalty = tf.contrib.layers.apply_regularization(regularizer, weights_list=var_list)

	# Op for calculating the loss
    with tf.name_scope("cross_ent"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y)) + penalty

	# Train op
    with tf.name_scope("train"):
		# Get gradients of all trainable variables
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))

		# Create optimizer and apply gradient descent to the trainable variables
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)

	'''
	# Add gradients to summary
	for gradient, var in gradients:
		tf.summary.histogram(var.name + '/gradient', gradient)

	# Add the variables we train to the summary
	for var in var_list:
		tf.summary.histogram(var.name, var)

	# Add the loss to summary
	tf.summary.scalar('cross_entropy', loss)
	'''

	# Evaluation op: Accuracy of the model
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	'''
	# Add the accuracy to the summary
	tf.summary.scalar('accuracy', accuracy)

	# Merge all summaries together
	merged_summary = tf.summary.merge_all()

	# Initialize the FileWriter
	writer = tf.summary.FileWriter(filewriter_path)

	# Initialize an saver for store model checkpoints
	saver = tf.train.Saver()
	'''
	# Initalize the data generator seperately for the training and validation set
    train_generator = ImageDataGenerator(train_file, num_classes, Input_channel, shuffle = True, horizontal_flip=False)
    val_generator = ImageDataGenerator(val_file, num_classes, Input_channel, shuffle = True)

	# Get the number of training/validation steps per epoch
    train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
    val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)


# Start Tensorflow session
config = tf.ConfigProto(allow_soft_placement = True)
with tf.Session(config=config) as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    #writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    for i in range(Num_ch):
        AlexNet[i].load_initial_weights(sess)


    print("{} Start training...".format(datetime.now()))
    #print("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))

    val_loss_array = []
    val_acc_array = []

    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("------------------------------------------")
        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        step = 1
        train_loss = 0.
        train_acc = 0.
        while step < train_batches_per_epoch:

            # Get a batch of images and labels
            batch_xs, batch_ys = train_generator.next_batch(batch_size)

            # And run the training op
            tr_loss, tr_acc, _ = sess.run([loss, accuracy, train_op],
                                           feed_dict = {x: batch_xs,
                                                        y: batch_ys,
                                                        keep_prob: dropout_rate})

            # Generate summary with the current batch of data and write to file
            '''
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: batch_xs,
                                                        y: batch_ys,
                                                        keep_prob: 1.})
                writer.add_summary(s, epoch * train_batches_per_epoch + step)
            '''

            train_loss += tr_loss/train_batches_per_epoch
            train_acc += tr_acc/train_batches_per_epoch

            step += 1


        print("{} Training Accuracy = {:.4f}".format(datetime.now(), train_acc))
        print("{} Training Loss = {:.4f}".format(datetime.now(), train_loss))
        print(sess.run(global_step))

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        val_loss = 0.
        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_generator.next_batch(batch_size)
            cost, acc = sess.run([loss, accuracy],
                                 feed_dict = {x: batch_tx, y: batch_ty, keep_prob: 1.})
            val_loss += cost
            test_acc += acc
            test_count += 1

        val_loss /= test_count
        test_acc /= test_count

        val_loss_array.append(val_loss)
        val_acc_array.append(test_acc)
        print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))
        print("{} Validation Loss = {:.4f}".format(datetime.now(), val_loss))
        print(sess.run(global_step))
        # Reset the file pointer of the image data generator
        val_generator.reset_pointer()
        train_generator.reset_pointer()

        '''
        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
        '''

    plt.figure(num='Loss&Acc')
    #plt.title("Loss & Acc\nlearning rate = %f" % lr)
    plt.title("Loss & Acc va")
    plt.plot(np.arange(1, num_epochs + 1), val_loss_array, label='Loss', linewidth=2)
    plt.plot(np.arange(1, num_epochs + 1), val_acc_array, label='Acc', linewidth=2)
    plt.xlabel('epochs')
    plt.legend(loc='best')
    plt.show()
