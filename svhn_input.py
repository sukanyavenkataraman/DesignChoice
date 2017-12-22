'''
SVHN input - present in .mat files. Loading it directly as an np array
'''

import scipy.io, numpy as np
import mrbi_input, cifar10_input
from matplotlib import pyplot as plt

# Width and height of each image.
img_size = 32

# Number of classes.
num_classes = 10

# Number of images in the training-set.
_num_images_train = 73257
_num_images_test = 26032

num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size*num_channels

def vgg_get_data(filename, num_files=7, num_images=73257):

    images = np.zeros(shape=[num_images, 4, 4, 256]) #TODO: remove this hard coding
    begin = 0

    for i in range(num_files):
        images_batch = np.load(filename+'_vgg_'+str(i+1))

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # The begin-index for the next batch is the current end-index.
        begin = end

    return images


def get_data(filename, is_vgg=False, train=True, one_hot_encoding=False):

    data = scipy.io.loadmat(filename)

    if is_vgg:
        images = vgg_get_data(filename, 7 if train else 2, _num_images_train if train else _num_images_test)
    else:
        images = data['X'].transpose(3,0,1,2)
        #print raw_float[0]
        print images.shape, data['y'].shape
        # Convert the raw images from the data-files to floating-points.

        images = np.array(images, dtype=float) / 255.0

    # Plot and check!
#    plt.imshow(images[0])
#    plt.show()

    cls = data['y'].reshape((-1))
    cls[cls == 10] = 0

    if one_hot_encoding:
        # TODO:How does next_batch affect one hot encoding?!
        return mrbi_input.DataSet(images, mrbi_input.dense_to_one_hot(cls, 10), channels=True)
    else:
        return mrbi_input.DataSet(images, cls)