# Get the weights from vgg-16 pretrained and train a network on top of it

import numpy as np
import tensorflow as tf

# TODO: Check if we need to do mean subtraction on inputs

class Vgg16:
    def __init__(self, filename='/home/sukanya/Downloads/vgg16.npy'):
        self.data_dict = np.load(filename, encoding='latin1').item()

    def get_vgg_conv_layers(self, input, num_conv_layers=5):

        name_conv_prefix = 'conv'
        name_pool_prefix = 'pool'

        input_layer = input
        for i in range(1,num_conv_layers+1):

            name_conv_outer = name_conv_prefix + str(i) + '_'
            name_pool = name_pool_prefix + str(i)

            if i in (1,2): # Only 2 conv layers
                conv1 = self.conv_layer(input_layer, name_conv_outer+'1')
                conv2 = self.conv_layer(conv1, name_conv_outer+'2')
                pool = self.max_pool(conv2, name_pool)

            if i in (3,4,5): # 3 conv layers
                conv1 = self.conv_layer(input_layer, name_conv_outer+'1')
                conv2 = self.conv_layer(conv1, name_conv_outer+'2')
                conv3 = self.conv_layer(conv2, name_conv_outer+'3')
                pool = self.max_pool(conv3, name_pool)

            input_layer = pool

        print input_layer.shape
        return input_layer

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

class AlexNet:
    def __init__(self, filename='/home/sukanya/Downloads/bvlc_alexnet.npy'):
        self.data_dict = np.load(filename).item()

        for key in self.data_dict:
            print key, ' ', self.data_dict[key][0].shape

    def conv(self, input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
        '''From https://github.com/ethereon/caffe-tensorflow
        '''
        c_i = input.get_shape()[-1]
        assert c_i % group == 0
        assert c_o % group == 0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

        if group == 1:
            conv = convolve(input, kernel)
        else:
            input_groups = tf.split(axis=group, num_or_size_splits=3, value=input)
            print len(input_groups)
            print input_groups[0].get_shape()
            print input_groups[1].get_shape()
            print input_groups[2].get_shape()
            kernel_groups = tf.split(axis=group, num_or_size_splits=3, value=kernel)
            print len(kernel_groups)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            conv = tf.concat(3, output_groups)
        b,h,w,c = conv.get_shape().as_list()
        print h,w,c
        return tf.reshape(tf.nn.bias_add(conv, biases), [-1,h,w,c])

    def get_alexnet_conv_layers(self, input, num_conv_layers=5):
        # conv1
        # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        k_h = 11;
        k_w = 11;
        c_o = 96;
        s_h = 4;
        s_w = 4
        conv1W = tf.Variable(self.data_dict["conv1"][0])
        conv1b = tf.Variable(self.data_dict["conv1"][1])
        conv1_in = self.conv(input, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding='SAME', group=1)
        conv1 = tf.nn.relu(conv1_in)

        # lrn1
        # lrn(2, 2e-05, 0.75, name='norm1')
        radius = 2;
        alpha = 2e-05;
        beta = 0.75;
        bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

        # maxpool1
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        k_h = 3;
        k_w = 3;
        s_h = 2;
        s_w = 2;
        padding = 'VALID'
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        print maxpool1.shape

        # conv2
        # conv(5, 5, 256, 1, 1, group=2, name='conv2')
        k_h = 5;
        k_w = 5;
        c_o = 256;
        s_h = 1;
        s_w = 1;
        group = 2
        conv2W = tf.Variable(self.data_dict["conv2"][0])
        print conv2W.get_shape()
        conv2b = tf.Variable(self.data_dict["conv2"][1])
        conv2_in = self.conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv2 = tf.nn.relu(conv2_in)

        # lrn2
        # lrn(2, 2e-05, 0.75, name='norm2')
        radius = 2;
        alpha = 2e-05;
        beta = 0.75;
        bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

        # maxpool2
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
        k_h = 3;
        k_w = 3;
        s_h = 2;
        s_w = 2;
        padding = 'VALID'
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        # conv3
        # conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3;
        k_w = 3;
        c_o = 384;
        s_h = 1;
        s_w = 1;
        group = 1
        conv3W = tf.Variable(self.data_dict["conv3"][0])
        conv3b = tf.Variable(self.data_dict["conv3"][1])
        conv3_in = self.conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv3 = tf.nn.relu(conv3_in)

        # conv4
        # conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3;
        k_w = 3;
        c_o = 384;
        s_h = 1;
        s_w = 1;
        group = 2
        conv4W = tf.Variable(self.data_dict["conv4"][0])
        conv4b = tf.Variable(self.data_dict["conv4"][1])
        conv4_in = self.conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv4 = tf.nn.relu(conv4_in)

        # conv5
        # conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3;
        k_w = 3;
        c_o = 256;
        s_h = 1;
        s_w = 1;
        group = 2
        conv5W = tf.Variable(self.data_dict["conv5"][0])
        conv5b = tf.Variable(self.data_dict["conv5"][1])
        conv5_in = self.conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv5 = tf.nn.relu(conv5_in)

        # maxpool5
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        k_h = 3;
        k_w = 3;
        s_h = 2;
        s_w = 2;
        padding = 'VALID'
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        return maxpool5
