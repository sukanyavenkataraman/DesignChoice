"""
This file contains the step by step implementation of a fully connected neural network of configurable depths,
hidden layer size, input, output, etc
"""

import tensorflow as tf, random, math, os
from tensorflow.examples.tutorials.mnist import input_data
import cifar10_input, mrbi_input, svhn_input
import scattering as scattering
from vgg_pretrained import Vgg16, AlexNet
import BrainImages_input
from matplotlib import pyplot as plt
import numpy as np

class Hyperparams:
    def __init__(self,
                 hidden_layer_sizes,
                 learning_rate): #Extend support for more if required

        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate

class DataSet:
    def __init__(self,
                 data_type='mnist',
                 use_vgg_pretrained = False,
                 num_conv_layers = 3,
		 kl_type='mean',
	         num_classes=4
                 ):

        self.vgg_pretrained = use_vgg_pretrained
        self.vgg_num_conv_layers = num_conv_layers
        self.data_type = data_type

        if data_type == 'mnist':
            mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
            self.next_batch = mnist.train.next_batch
            self.eval_images = mnist.test.images
            self.eval_labels = mnist.test.labels
            self.input_dim = 28
            self.num_classes = 10
            self.num_channels = 1
            self.num_images = len(mnist.train.images)
            self.scat_w_filterh = 1
            self.scat_w_filterw = 1
	    self.size = self.eval_images.shape[-1]
	    print self.num_images

        if data_type == 'cifar10':
            cifar_dir = '/home/sukanya/Documents/'
            cifar10 = cifar10_input.load_training_data(cifar_dir, self.vgg_pretrained, one_hot_encoding=True)
            self.next_batch = cifar10.next_batch
            eval = cifar10_input.load_test_data(cifar_dir, self.vgg_pretrained, one_hot_encoding=True)
            self.eval_images = eval.images
            self.eval_labels = eval.labels
            self.input_dim = 32
            self.num_classes = 10
            self.num_channels = 3
            self.num_images = len(cifar10.images)
            self.scat_w_filterh = 5
            self.scat_w_filterw = 5
	    self.size = self.eval_images.shape[-1]
            print cifar10.images.shape

        if data_type == 'mrbi':
            mrbi_dir = '/home/sukanya/Documents/mrbi/'
            mrbi_train_file = mrbi_dir + 'mrbi_train.amat'
            mrbi_test_file = mrbi_dir + 'mrbi_test.amat'

            mrbi = mrbi_input.get_data(mrbi_train_file, train=True, one_hot_encoding=True, do_scattering_transform=False,
                                       J=2)
            self.next_batch = mrbi.next_batch
            eval = mrbi_input.get_data(mrbi_test_file, train=False, one_hot_encoding=True)
            self.eval_images = eval.images
            self.eval_labels = eval.labels
            self.input_dim = 28
            self.num_classes = 10
            self.num_channels = 1
            self.num_images = len(mrbi.images)
	    self.size = self.eval_images.shape[-1]
            self.scat_w_filterh = 5
            self.scat_w_filterw = 5

        if data_type == 'svhn':
            svhn_dir = '/home/sukanya/Documents/SVHN/'
            svhn_train_file = svhn_dir + 'train_32x32.mat'
            svhn_test_file = svhn_dir + 'test_32x32.mat'

            svhn = svhn_input.get_data(svhn_train_file, is_vgg=self.vgg_pretrained, train=True, one_hot_encoding=True)
            self.next_batch = svhn.next_batch
            eval = svhn_input.get_data(svhn_test_file, is_vgg=self.vgg_pretrained, train=False, one_hot_encoding=True)
            self.eval_images = eval.images
            self.eval_labels = eval.labels
            self.input_dim = 32
            self.num_classes = 10
            self.num_channels = 3
            self.num_images = len(svhn.images)
            self.scat_w_filterh = 5
            self.scat_w_filterw = 5
	    self.size = self.eval_images.shape[-1]
            print svhn.images.shape

        if data_type == 'fdg' or data_type == 'av45' or data_type == 'mri':
            train, eval = BrainImages_input.get_data(one_hot_encoding=True, intype=data_type, kl_type=kl_type, num_classes=num_classes)
            self.next_batch = train.next_batch
            self.eval_images = eval.images
            self.eval_labels = eval.labels
            self.input_dim = 79  # Works ONLY because downstream it is used as input_dim*input_dim*num_channels
            self.num_classes = 4
            self.num_channels = 95
            self.num_images = len(train.images)
	    self.size = self.eval_images.shape[-1]
            print train.images.shape


class deepNN:
    def __init__(self,
                 input_data,
                 random_hls_gen = False,
                 hls_low = 1,
                 hls_high = 500,
                 hidden_layer_sizes = [],
                 custom_hls_distr = False,
                 hls_distr = {},
                 num_hidden_layers=2,
                 batch_size=784,
                 do_batch_norm=True,
                 random_lr_gen = False,
                 lr_lowr = 1e-3,
                 lr_highr = 1e-1,
                 learning_rate=1e-3,
                 activation_fn = 'sigmoid',
                 scattering_transform = False,
                 alexnet_pretrained=False,
                 alexnet_num_conv_layers=5,
                 save_models = False,
		 model_dir = '/home/sukanya/PycharmProjects/TensorFlow/Hyperband/Models_Medical/',
                 model_name='HB',
                 decay_rate=0.99,
                 decay_steps=1000):

        # Flags
        self.random_hls_gen = random_hls_gen
        self.custom_hls_distr = custom_hls_distr
        self.random_lr_gen = random_lr_gen
        self.do_batch_norm = do_batch_norm
        self.do_scat_transform = scattering_transform
        self.use_vgg_pretrained = input_data.vgg_pretrained
        self.use_alexnet_pretrained = alexnet_pretrained
        self.save_models = save_models

        # HLS specific variables
        self.hls_low = hls_low
        self.hls_high = hls_high
        self.hls_cdf = hls_distr
        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_hl = num_hidden_layers

        # LR specific variables
        self.lr_lowr = lr_lowr
        self.lr_highr = lr_highr
        self.learning_rate = learning_rate

        # Network specific variables
        self.num_hidden_layers = num_hidden_layers
        self.batch_size = batch_size
        self.activation_fn = activation_fn

        # Vgg pretrained specific variables
        self.vgg_pt_filename = '/home/sukanya/Downloads/vgg16.npy'
        self.vgg_num_conv_layers = input_data.vgg_num_conv_layers

        # Alexnet pretrained specific variables
        self.alexnet_pt_filename = '/home/sukanya/Downloads/bvlc_alexnet.npy'
        self.alexnet_num_conv_layers = alexnet_num_conv_layers

        # Save all the models to use while running hyperband
        self.models = {}
        self.saved_model_loc = model_dir+input_data.data_type+'/'+model_name+'/'
	if not os.path.exists(model_dir):
	    os.mkdir(model_dir)
	if not os.path.exists(model_dir+input_data.data_type):
	    os.mkdir(model_dir+input_data.data_type)
        if not os.path.exists(self.saved_model_loc):
            os.mkdir(self.saved_model_loc)

        self.global_count = 0

        # Decay variables
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

        # Input data specific variables
        self.next_batch = input_data.next_batch
        self.eval_images = input_data.eval_images
        self.eval_labels = input_data.eval_labels
        self.num_classes = input_data.num_classes
        self.input_dim = input_data.input_dim
        self.num_channels = input_data.num_channels
        self.train_num_batches = input_data.num_images/self.batch_size
        self.img_size = input_data.size

    # Below two functions are taken from the tutorial
    def weight_variable(self, shape, stddev=0.1):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial, name='w')

    def bias_variable(self, shape, constant=0.1):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(constant, shape=shape)
        return tf.Variable(initial, name='b')

    def add_layer(self, input, hidden_layer_size_op):
         prev_len = input.shape[-1].value
         w = self.weight_variable([prev_len, hidden_layer_size_op])
         b = self.bias_variable([hidden_layer_size_op])

         output = tf.matmul(input, w) + b

         if self.do_batch_norm:
             mean, variance = tf.nn.moments(output, [0,1]) # Check for axis here!
             scale = tf.Variable(tf.ones([hidden_layer_size_op]))
             beta = tf.Variable(tf.zeros([hidden_layer_size_op]))

             output = tf.nn.batch_normalization(output, mean=mean, variance=variance, scale=scale, offset=beta, variance_epsilon=1e-9)

         if self.activation_fn == 'sigmoid':
             h = tf.nn.sigmoid(output)
         if self.activation_fn == 'relu':
             h = tf.nn.relu(output)
         if self.activation_fn == 'crelu':
             h = tf.nn.softplus(output)

         return w, b, h

    def construct_network(self, x, hidden_layer_sizes):

        name_prefix = 'fc'
        keep_prob = tf.placeholder(tf.float32)

    # Scattering transform on input x : This is in channel first format. Make sure data is channel first!

        if self.do_scat_transform:
            x = tf.reshape(x, [-1, self.num_channels, self.input_dim, self.input_dim])
            print x.shape
            M, N = x.get_shape().as_list()[-2:]
            print M,N
            x = scattering.Scattering(M=self.input_dim, N=self.input_dim, J=2)(x) # Experiment with multiple J?
            print x.get_shape
            x_c, x_h, x_w = x.get_shape().as_list()[-3:]
            print x_c, x_h, x_w
            x = tf.reshape(x, (-1,x_h,x_w,x_c)) # Reshaping for CPU spported NHWC

            x = tf.contrib.layers.batch_norm(x, fused=True)

            w = self.weight_variable([self.scat_w_filterh, self.scat_w_filterw, x_c, 64], stddev=0.02) # Remove hard coding later on
            b = self.bias_variable([64])

            conv_op = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding="SAME")

            # Add bias - CPU supports only NHWC. Rearrange?
            conv_op = tf.reshape(tf.nn.bias_add(conv_op, b), (-1,x_h,x_w,64))

            target_shape = (-1, 64 * x_h * x_w)
            x = tf.reshape(conv_op, target_shape)

        #if self.use_vgg_pretrained:
            #vgg16 = Vgg16(self.vgg_pt_filename)
            #x = tf.reshape(x, [-1, self.input_dim, self.input_dim, self.num_channels])
            #x = vgg16.get_vgg_conv_layers(x, self.vgg_num_conv_layers)
            #shape = x.get_shape().as_list()
            #dim = 1
            #for d in shape[1:]:
            #    dim *= d
            #x = tf.reshape(x, [-1, dim])

        if self.use_alexnet_pretrained:
            alexnet = AlexNet(self.alexnet_pt_filename)
            x = tf.reshape(x, [-1, self.input_dim, self.input_dim, self.num_channels])
            x = alexnet.get_alexnet_conv_layers(x, self.alexnet_num_conv_layers)
            shape = x.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(x, [-1, dim])
            print x.shape

        for i in range(len(hidden_layer_sizes)):
            name = name_prefix + str(i)

            with tf.name_scope(name):
                curr_len = int(hidden_layer_sizes[i])
                w, b, h = self.add_layer(x if i==0 else input, curr_len)
                print w.shape, b.shape, h.shape

            with tf.name_scope('dropout'):
                h2 = tf.nn.dropout(h, keep_prob)

            input = h2

        with tf.name_scope('final_layer'):
            w = self.weight_variable([curr_len, self.num_classes])
            b = self.bias_variable([self.num_classes])

            y = tf.matmul(h2, w) + b

        y_softmax = tf.nn.softmax(y, name = 'softmax')

        return y_softmax, keep_prob, h2

    def run_eval_hyperparam_withbs(self, hyperparams, num_epochs, keepProb):

        tf.reset_default_graph() # To enable restoring and reloading saved files
	
        # Input data
        if self.use_vgg_pretrained:
            x = tf.placeholder(tf.float32, [None, 4*4*256]) # TODO: Remove hard coding!
        else:
            x = tf.placeholder(tf.float32, [None, self.img_size])

        # Output data
        y = tf.placeholder(tf.float32, [None, self.num_classes])

        # Graph for neural net
        y_nn, keep_prob, h2 = self.construct_network(x, hyperparams.hidden_layer_sizes)

        # TODO: Check if below is necessary!
        with tf.name_scope('loss'):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_nn)

        cross_entropy = tf.reduce_mean(cross_entropy)

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(hyperparams.learning_rate, global_step, self.decay_steps, self.decay_rate, staircase=True)

        with tf.name_scope('optimiser'):
            train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy, global_step=global_step)

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(tf.argmax(y_nn, 1), tf.argmax(y, 1))
            correct_predictions = tf.cast(correct_predictions, tf.float32)

        accuracy = tf.reduce_mean(correct_predictions)

        if self.save_models:
            saver = tf.train.Saver()

	#accuracy_summary = tf.summary.scalar("Training Accuracy", accuracy)
	#summaries_dir = '/home/sukanya/PycharmProjects/TensorFlow/Hyperband/Summaries/'

        with tf.Session() as sess:
	    #train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
	    #test_writer = tf.summary.FileWriter(summaries_dir + '/test')
            sess.run(tf.global_variables_initializer())

            if self.save_models:
                if hyperparams in self.models:
                    print 'Restoring from saved model!'
                    model_location = self.models[hyperparams]
                    saver.restore(sess, model_location)
                else:
                    model_location = self.saved_model_loc + str(self.global_count) + '_'+str(hyperparams)+'/model.ckpt'
                    self.models[hyperparams] = model_location
                    self.global_count += 1

            for epoch in range(num_epochs):
                for i in range(self.train_num_batches):
                    batch = self.next_batch(self.batch_size)

                    if i % 20 == 0:
                        train_accuracy = accuracy.eval(feed_dict={
                            x: batch[0], y: batch[1], keep_prob: 1.0})
                        print('epoch %d step %d global step %d, training accuracy %g learning_rate %f' % (epoch, i, global_step.eval(), train_accuracy, learning_rate.eval()))
		    #print batch[0]
                    #summary, step = sess.run([accuracy_summary, global_step], feed_dict={x: batch[0], y: batch[1], keep_prob: keepProb})
		    #train_writer.add_summary(summary, epoch)
		
                #evaluation = accuracy.eval(feed_dict={x: self.eval_images, y: self.eval_labels, keep_prob: 1.0})
                    train.run(feed_dict={x: batch[0], y: batch[1], keep_prob: keepProb})
                y_h = sess.run([accuracy, cross_entropy, correct_predictions, h2], feed_dict={x: self.eval_images, y: self.eval_labels, keep_prob: 1.0})
	#	test_writer.add_summary(y_h[0], epoch)

                '''
                print y_h[3].shape
                to_plot = y_h[3]
                print  to_plot
                to_plot_np = np.asarray(to_plot)

                to_plot = np.reshape(to_plot_np, [-1,1,10,10])
                print to_plot

                plt.imshow(to_plot[0][0])
                plt.show()

                plt.imshow(to_plot[0][0])
                plt.show()
                '''
                print ('Evaluation after epoch %d is : %f' %(epoch, y_h[0]))
	    
            if self.save_models:
                save_path = saver.save(sess, model_location)
                print 'model saved in - ', save_path
            return 1.0 - y_h[0]

    # Generate the hyperparam space - currently for learning rate, hidden layer sizes
    def get_from_hyperparam_space(self, base=10):

        if self.random_lr_gen:
            rand_lr = math.log(random.uniform(base ** self.lr_lowr, base ** self.lr_highr), base)
        else:
            rand_lr = self.learning_rate

        hls = []

        if self.random_hls_gen:
            if self.custom_hls_distr:
                hls = self.hls_cdf.get(float(min(self.hls_cdf, key=lambda x: abs(x - random.uniform(0, 1))))) # Change this when running main.py!
                '''
                sizes = hls_string.split(' ')

                for k in range(len(sizes)):
                    hls.append(sizes[k].strip())
                '''
                print(hls)

            else:
                for i in range(self.num_hl):
                    hls.append(random.randint(self.hls_low, self.hls_high))

        else:
            size = len(self.hidden_layer_size)
            index = random.randint(0, size - 1)
            hls = self.hidden_layer_size[index]

        h = Hyperparams(hls, rand_lr)
        return h










