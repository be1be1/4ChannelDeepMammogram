import tensorflow as tf
import numpy as np

class Simple_CNN(object):

    def __init__(self, x, num_classes, num_ch, ch_idx, skip_layer, weights_path='DEFAULT'):
        self.X = x
        self.NUM_CLASSES = num_classes
        self.NUM_CH = num_ch
        self.CH_IDX = ch_idx

        self.SKIP_LAYER = skip_layer

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        self.create()

    def create(self):

        conv1 = conv(self.X, self.CH_IDX, 3, 3, 16, 1, 1, padding='SAME', name='conv1')
        pool1 = max_pool(conv1, 2, 2, 2, 2, padding='VALID', name='pool1')
        conv2 = conv(pool1, self.CH_IDX, 3, 3, 32, 1, 1, padding='SAME', name='conv2')
        pool2 = max_pool(conv2, 2, 2, 2, 2, padding='VALID', name='pool2')
        conv3 = conv(pool2, self.CH_IDX, 3, 3, 64, 1, 1, padding='SAME', name='conv3')
        pool3 = max_pool(conv3, 2, 2, 2, 2, padding='VALID', name='pool3')
        conv4 = conv(pool3, self.CH_IDX, 3, 3, 128, 1, 1, padding='SAME', name='conv4')
        pool4 = max_pool(conv4, 2, 2, 2, 2, padding='VALID', name='pool4')
        conv5 = conv(pool4, self.CH_IDX, 3, 3, 128, 1, 1, padding='SAME', name='conv5')
        pool5 = max_pool(conv5, 2, 2, 2, 2, padding='VALID', name='pool5')

        flattened = tf.reshape(pool5, [-1, 8*8*128])
        fc6 = fc(flattened, self.CH_IDX, 8*8*128, int(4096 / self.NUM_CH), name='fc6')
        self.fc7 = fc(fc6, 0, 4096, self.NUM_CLASSES, relu=True, name='fc7')


    def load_initial_weights(self, session):
        """
        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ come
        as a dict of lists (e.g. weights['conv1'] is a list) and not as dict of
        dicts (e.g. weights['conv1'] is a dict with keys 'weights' & 'biases') we
        need a special load function
        """

        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if the layer is one of the layers that should be reinitialized
            if op_name not in self.SKIP_LAYER:
                with tf.variable_scope(op_name, reuse=True):

                    # Loop over list of weights/biases and assign them to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases'+str(self.CH_IDX), trainable=False)
                            session.run(var.assign(data))

                        # Weights
                        else:
                            var = tf.get_variable('weights'+str(self.CH_IDX), trainable=False)
                            session.run(var.assign(data))


#################################################################################
def conv(x, channel, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):

    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights'+str(channel), shape=[filter_height, filter_width, input_channels / groups, num_filters])
        biases = tf.get_variable('biases'+str(channel), shape=[num_filters])

        if groups == 1:
            conv = convolve(x, weights)

        # In the cases of multiple groups, split inputs & weights and
        else:

            # Split input and weights and convolve them separately
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(axis=3, values=output_groups)

        # Add biases
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)

    return relu




def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)



def fc(x, channel, num_in, num_out, name, relu=True):
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights'+str(channel), shape=[num_in, num_out], trainable=True)
        biases = tf.get_variable('biases'+str(channel), [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu

        else:
            return act