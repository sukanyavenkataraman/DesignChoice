
import numpy as np
import cPickle
import os
import mrbi_input
from matplotlib import pyplot as plt

# Width and height of each image.
img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 10

# Number of files for the training-set.
_num_files_train = 5

# Number of images for each batch-file in the training-set.
_images_per_file = 10000

# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.
_num_images_train = _num_files_train * _images_per_file

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot

def _unpickle(data_dir, filename):
    # Create full path for the file.
    file_path = os.path.join(data_dir, "cifar-10-batches-py/%s"%filename)

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
         data = cPickle.load(file)

    return data


def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0
    print raw_float.shape

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


def _load_data(data_dir, filename):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images (see above) and the class-number
    for each image.
    """

    # Load the pickled data-file.
    data = _unpickle(data_dir, filename)

    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'labels'])

    # Convert the images.
    images = _convert_images(raw_images)

    return images, cls

def load_class_names(data_dir):
    """
    Load the names for the classes in the CIFAR-10 data-set.
    Returns a list with the names. Example: names[3] is the name
    associated with class-number 3.
    """

    # Load the class-names from the pickled file.
    raw = _unpickle(data_dir, filename="batches.meta")[b'label_names']

    # Convert from binary strings.
    names = [x.decode('utf-8') for x in raw]

    return names

def vgg_get_data(data_dir, filename):
    images = np.load(data_dir + "cifar-10-batches-py/%s"%filename)
    return images

def load_training_data(data_dir, vgg=False, one_hot_encoding=False):
    """
    Load all the training-data for the CIFAR-10 data-set.
    The data-set is split into 5 data-files which are merged here.
    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[_num_images_train, 4 if vgg else img_size, 4 if vgg else img_size, 256 if vgg else num_channels], dtype=float) # TODO: Remove hard coding!
    cls = np.zeros(shape=[_num_images_train], dtype=int)

    # Begin-index for the current batch.
    begin = 0
    # For each data-file.
    for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = _load_data(data_dir, filename="data_batch_" + str(i + 1))

        if vgg:
            images_batch_vgg = vgg_get_data(data_dir, filename="vgg_data_batch_" + str(i + 1))
 
        # Number of images in this batch.
        num_images = len(images_batch) if not vgg else len(images_batch_vgg)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch if not vgg else images_batch_vgg

        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch

        # The begin-index for the next batch is the current end-index.
        begin = end


    if one_hot_encoding:
        return mrbi_input.DataSet(images, dense_to_one_hot(cls, 10), channels=True)
    else:
        return mrbi_input.DataSet(images, cls, channels=True)


def load_test_data(data_dir, vgg = False, one_hot_encoding=False):
    """
    Load all the test-data for the CIFAR-10 data-set.
    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    images, cls = _load_data(data_dir, filename="test_batch")

    if vgg:
	images_vgg = vgg_get_data(data_dir, filename="vgg_test_data")

    if one_hot_encoding:
        return mrbi_input.DataSet(images if not vgg else images_vgg, dense_to_one_hot(cls, 10), channels=True)
    else:
        return mrbi_input.DataSet(images if not vgg else images_vgg, cls, channels=True)
