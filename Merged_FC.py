import tensorflow as tf

class Merged_FC(object):

    def __init__(self, input, keep_prob, num_classes):

        self.INPUT = input
        self.KEEP_PROB = keep_prob
        self.NUM_CLASSES = num_classes
        self.create()

    def create(self):
        dropout6 = dropout(self.INPUT, self.KEEP_PROB)
        fc7 = fc(dropout6, 0, 4096, 4096, name = 'fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)
        self.fc8 = fc(dropout7, 0, 4096, self.NUM_CLASSES, relu=False, name='fc8')

####################################################################################

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


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)
