
import tensorflow as tf
import vgg_pretrained
import cifar10_input, svhn_input, math
import numpy as np
# Change images to output of vgg pretrained model


def change_ip(input, num_conv_layers, filename='/home/sukanya/Documents/cifar10_vgg/vgg_test_batch', img_size=32):
    tf.reset_default_graph()  # To enable restoring and reloading saved files

    f = open(filename, 'w')

    #input = np.asarray(input)
    # Input data
    x = tf.convert_to_tensor(input, np.float32)
    x = tf.reshape(x, [-1, img_size, img_size, 3])
    vgg16 = vgg_pretrained.Vgg16()
    conv_op = vgg16.get_vgg_conv_layers(x, num_conv_layers)

    sess = tf.InteractiveSession()
    #print conv_op.eval()
    np.save(f, conv_op.eval(), allow_pickle=True)
    f.close()
    #f.seek(0)

    #op = np.load('just_checking')

#    print op

   # print conv_op.eval()
'''
# cifar10
cifar_dir = '/home/sukanya/Documents/'
cifar10 = cifar10_input.load_test_data(cifar_dir, one_hot_encoding=True)

prev = 0
for i in range(1):
    change_ip(cifar10.images[0+prev:prev+10000], 3, i+1)
    prev += 10000
'''
#svhn
_num_images_train = 73257
_num_images_test = 26032
num_channels = 3
num_layers_used = 3
num_iter_train = 7
num_iter_test = 2
svhn_dir = '/home/sukanya/Documents/SVHN/'
svhn_train_file = svhn_dir + 'train_32x32.mat'
svhn_test_file = svhn_dir + 'test_32x32.mat'

svhn_train = svhn_input.get_data(svhn_train_file, train=True, one_hot_encoding=True)
svhn_test = svhn_input.get_data(svhn_test_file, train=False, one_hot_encoding=True)

prev = 0

for i in range(num_iter_train):
    change_ip(svhn_train.images[prev:prev+10465], num_layers_used, filename=svhn_train_file+'_vgg_'+str(i+1), img_size=32)
    prev += 10465

prev=0
for i in range(num_iter_test):
    change_ip(svhn_test.images[prev:prev+13016], num_layers_used, filename=svhn_test_file+'_vgg_'+str(i+1), img_size=32)
    prev += 13016


