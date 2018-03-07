import numpy as np
import imp
import scattering as scattering
from sklearn.model_selection import StratifiedShuffleSplit
#TODO: Make all input reads the same
random_seed = imp.load_source('random_seed', '/home/sukanya/tensorflow/local/lib/python2.7/site-packages/tensorflow/python/framework/random_seed.py')

# Number of classes.
num_classes = 10

# Number of images for each batch-file in the training-set.
_num_images_train = 12000
_num_images_test = 50000

num_channels = 1

class DataSet(object):

  def __init__(self,
               images,
               labels,
               dtype=np.float32,
               reshape=True,
               channels = True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    np.random.seed(seed1 if seed is None else seed2)

    assert images.shape[0] == labels.shape[0], (
         'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
    self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    if reshape:
      if channels:
          images = images.reshape(images.shape[0],
                                  images.shape[1] * images.shape[2] * images.shape[3])
      else:
          images = images.reshape(images.shape[0],
                                  images.shape[1] * images.shape[2])

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""

    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    labels_one_hot = labels_one_hot.reshape([num_labels, num_classes])
    return labels_one_hot


def get_data(filename, train = True, one_hot_encoding=False, do_scattering_transform=False, J=2, img_size=28): # One hot encoding is generally false

    img_size_flat = img_size*img_size
    if do_scattering_transform:
        images = np.zeros(shape=[_num_images_train if train else _num_images_test, num_channels, img_size, img_size], dtype=float)
    else:
        images = np.zeros(shape=[_num_images_train if train else _num_images_test, img_size, img_size], dtype=float)

    cls = np.zeros(shape=[_num_images_train if train else _num_images_test], dtype=int)

    with open(filename, 'r') as f:
        lines = f.readlines()

    f.close()
    current_instance = 0

    for eachline in lines:

        parts = eachline.split(' ')

        parts = filter(None, parts)

        for i in range(len(parts)):
            parts[i] = float(parts[i].strip())

        parts = np.array(parts, dtype=float)
        if do_scattering_transform:
            x = np.reshape(parts[0:img_size_flat], [-1, num_channels, img_size, img_size])
            x = scattering.Scattering(M=img_size, N=img_size, J=J)(x)
            images[current_instance:current_instance + 1, :] = x

        else:
            parts_images = parts[0:img_size_flat].reshape([-1, img_size, img_size])
            images[current_instance:current_instance + 1, :] = parts_images

        parts_labels = parts[img_size_flat]
        cls[current_instance] = parts_labels

        current_instance += 1
    shuffleSplit = StratifiedShuffleSplit(n_splits=1, test_size=2000, random_state=np.random.RandomState())

    for train, valid in shuffleSplit.split(X=images, y=cls):
        train_set_images, train_set_labels = np.take(images, train, axis=0), np.take(cls, train, axis=0)
        valid_set_images, valid_set_labels = np.take(images, valid, axis=0), np.take(cls, valid, axis=0)

    if one_hot_encoding:
        return DataSet(train_set_images, dense_to_one_hot(train_set_labels, 10), channels=True) if do_scattering_transform else DataSet(images, dense_to_one_hot(cls, 10), channels=False),\
               DataSet(valid_set_images, dense_to_one_hot(valid_set_labels, 10), channels=False)
    else:
        return DataSet(images, cls, channels=False)


