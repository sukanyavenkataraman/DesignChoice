# A fully connected sigmoid neural network
# No. of hidden layers - 2
# Length of each hidden layer - Passed as a param
# Train, test data - MNIST (Later change this to be configurable)
# Activation function type - sigmoid
# Optimisation - minibatch gradient descent (Small batch size?)
# Loss function - Squared loss (check)
# Ends with softmax linear

# Also define two functions - get_from_hyperparam_space which returns hyperparams randomly chosen
# and run_eval_hyperparam which runs this model for a given num of epochs and evaluates it as well

# MNIST specific data -
# Num of output classes - 10 (0 to 9)
OP_SIZE = 10

# MNIST image sizes are 28x28
MNIST_HL_DIM = 28*28

import tensorflow as tf, random, math
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import cifar10_input, mrbi_input, svhn_input

class Hyperparams:
    def __init__(self,
                 hidden_layer_sizes,
                 learning_rate): #Extend support for more if required

        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate


class SigmoidNN:

    def __init__(self, hidden_layer_sizes = [],
                 learning_rate=1e-3,
                 random_hls_gen = False,
                 hls_low = 1,
                 hls_high = 500,
                 custom_hls_distr = False,
                 hls_distr = {},
                 num_hidden_layers=2,
                 data_type='MNIST',
                 batch_size=784,
                 random_lr_gen = False,
                 lr_lowr = 1e-3,
                 lr_highr = 1e-1,
                 activation_fn = 'sigmoid'):

            self.data_type = data_type
            self.num_hl = num_hidden_layers
            self.hidden_layer_size = hidden_layer_sizes
            self.hls_low = hls_low
            self.hls_high = hls_high
            self.lr_lowr = lr_lowr # lowest value learning_rate can take
            self.lr_highr = lr_highr # highest ""
            self.estimators = {}
            self.random_hls_gen = random_hls_gen
            self.random_lr_gen = random_lr_gen
            self.learning_rate = learning_rate
            self.custom_hls_distr = custom_hls_distr
            self.hls_cdf = hls_distr
            self.batch_size = batch_size

            if data_type == 'mnist':
                self.mnist_data = input_data.read_data_sets('MNIST_data', one_hot=False)
                #self.feature_columns = learn.infer_real_valued_columns_from_input(self.mnist_data.train.images)
                self.feature_columns = [tf.feature_column.numeric_column("x", shape=[784])] #Check this!
                self.train_inpfn = tf.estimator.inputs.numpy_input_fn(
                                            x={"x" : np.array(self.mnist_data.train.images)},
                                            y = np.asarray(self.mnist_data.train.labels, dtype=np.int32),
                                            batch_size=self.batch_size,
                                            shuffle=True)
                self.train_num_batches = len(self.mnist_data.train.images)/self.batch_size
                print(len(self.mnist_data.train.images))
                print(self.train_num_batches)

                self.eval_inpfn = tf.estimator.inputs.numpy_input_fn(
                                            x={"x" : np.array(self.mnist_data.validation.images)},
                                            y = np.asarray(self.mnist_data.validation.labels, dtype=np.int32),
                                            batch_size=self.batch_size,
                                            shuffle=True)
                self.eval_num_batches = len(self.mnist_data.validation.images)/self.batch_size

            if data_type == 'cifar10':
                cifar_dir = '/home/sukanya/Documents/' #Change this later to download and use?
                cifar_train = cifar10_input.load_training_data(cifar_dir)

                self.feature_columns = [tf.feature_column.numeric_column("x", shape=[3072])]  # Check this!
                self.train_inpfn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": cifar_train.images},
                    y = cifar_train.labels,
                    batch_size=self.batch_size,
                    shuffle=True)
                self.train_num_batches = len(cifar_train.images) / self.batch_size
                cifar_test = cifar10_input.load_test_data(cifar_dir)
                self.eval_inpfn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": cifar_test.images},
                    y = cifar_test.labels,
                    batch_size=self.batch_size,
                    shuffle=True)
                self.eval_num_batches = len(cifar_test.images) / self.batch_size

            if data_type == 'mrbi':
                mrbi_dir = '/home/sukanya/Documents/mrbi/'
                mrbi_train_file = mrbi_dir + 'mrbi_train.amat'
                mrbi_test_file = mrbi_dir + 'mrbi_test.amat'

                mrbi_train = mrbi_input.get_data(mrbi_train_file, train=True)
                self.feature_columns = [tf.feature_column.numeric_column("x", shape=[784])]  # Check this!
                self.train_inpfn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": mrbi_train.images},
                    y=mrbi_train.labels,
                    batch_size=self.batch_size,
                    shuffle=True)
                self.train_num_batches = len(mrbi_train.images) / self.batch_size

                print(len(mrbi_train.images))
                print(self.train_num_batches)

                mrbi_test = mrbi_input.get_data(mrbi_test_file, train=False)
                self.eval_inpfn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": mrbi_test.images},
                    y=mrbi_test.labels,
                    batch_size=self.batch_size,
                    shuffle=True)
                self.eval_num_batches = len(mrbi_test.images) / self.batch_size

                print(len(mrbi_test.images))

            if data_type == 'svhn':
                svhn_dir = '/home/sukanya/Documents/SVHN/'
                svhn_train_file = svhn_dir + 'train_32x32.mat'
                svhn_test_file = svhn_dir + 'test_32x32.mat'

                svhn_train = svhn_input.get_data(svhn_train_file, train=True)
                self.feature_columns = [tf.feature_column.numeric_column("x", shape=[3072])]  # Check this!
                self.train_inpfn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": svhn_train.images},
                    y=svhn_train.labels,
                    batch_size=self.batch_size,
                    shuffle=True)
                self.train_num_batches = len(svhn_train.images) / self.batch_size

                print(len(svhn_train.images))
                print(self.train_num_batches)

                svhn_test = svhn_input.get_data(svhn_test_file, train=False)
                self.eval_inpfn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": svhn_test.images},
                    y=svhn_test.labels,
                    batch_size=self.batch_size,
                    shuffle=True)
                self.eval_num_batches = len(svhn_test.images) / self.batch_size

                print(len(svhn_test.images))

            if activation_fn == 'sigmoid':
                self.activation_fn = tf.nn.sigmoid

            if activation_fn == 'relu':
                self.activation_fn = tf.nn.relu

            if activation_fn == 'crelu':
                self.activation_fn = tf.nn.softplus

    # With batch size
    def run_eval_hyperparam_withbs(self, hyperparams, num_epochs, dropout=0.25):

        if hyperparams in self.estimators:
            estimator = self.estimators.get(hyperparams)
        else:
            estimator = tf.estimator.DNNClassifier(hidden_units=hyperparams.hidden_layer_sizes,
                                                   feature_columns=self.feature_columns,
                                                   n_classes=OP_SIZE,
                                                   optimizer=tf.train.GradientDescentOptimizer(
                                                   learning_rate=hyperparams.learning_rate),#.minimize(
                                                            #loss=tf.losses.mean_squared_error),
                                                   activation_fn=self.activation_fn,
                                                   dropout=dropout) # Loss function??
            self.estimators[hyperparams] = estimator

        for i in range(num_epochs):
            estimator.train(input_fn=self.train_inpfn, steps=self.train_num_batches) # One iteration is over all batches
        evaluation = estimator.evaluate(input_fn=self.eval_inpfn, steps=self.eval_num_batches)
        # TODO: save gradients/norm of the gradients as well when training

        predictions = estimator.predict(input_fn=self.eval_inpfn)

        classes = {}
        for predict in predictions:
            if predict['classes'][0] in classes:
                classes[predict['classes'][0]] += 1
            else:
                classes[predict['classes'][0]] = 1

        print(classes)

        print(evaluation)
        return (1 - evaluation["accuracy"])


    # Get random values for the learning rate
    def get_from_hyperparam_space(self, base=10):

        if self.random_lr_gen:
            rand_lr = math.log(random.uniform(base**self.lr_lowr, base**self.lr_highr), base)
        else:
            rand_lr = self.learning_rate

        hls = []

        if self.random_hls_gen:
            if self.custom_hls_distr:
                hls_string = self.hls_cdf.get(float(min(self.hls_cdf, key=lambda x:abs(x-random.uniform(0,1)))))

                sizes = hls_string.split(' ')

                for k in range(len(sizes)):
                    hls.append(sizes[k].strip())

            else:
                for i in range(self.num_hl):
                    hls.append(random.randint(self.hls_low, self.hls_high))

        else:
            size = len(self.hidden_layer_size)
            index = random.randint(0, size - 1)
            hls = self.hidden_layer_size[index]

        print hls
        h = Hyperparams(hls, rand_lr)
        return h







