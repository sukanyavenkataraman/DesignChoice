# MRI, FDG and AV45 PET Scans
import numpy as np, scipy.io, h5py, scipy.stats, mrbi_input
from matplotlib import pyplot as plt
import time
import hdf5storage

base_dir = '/media/vamsi/datadrive2tb/'
fdg_filename = base_dir+'FDG_3visits/tp1_fdgs/fdgs_tp1_'
av45_filename = base_dir+'AV45_3visits/tp1_av45s/av45s_tp1_'
mri_filename = base_dir+'MRIs_3visits/tp1_mris/mris_tp1_'

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def get_kl_divergence(images_flattened, labels, intype):

     pixels = images_flattened.shape[-1]

     images_1 = []
     images_2 = []
     images_3 = []
     images_4 = []

     for i in range(len(images_flattened)):
        if labels[i] == 1:
            images_1.append(images_flattened[i:i+1,:])
        if labels[i] == 2:
            images_2.append(images_flattened[i:i+1,:])
        if labels[i] == 3:
            images_3.append(images_flattened[i:i+1,:])
        if labels[i] == 4:
            images_4.append(images_flattened[i:i+1,:])

     images_1 = np.reshape(np.asarray(images_1), [len(images_1), pixels])
     images_2 = np.reshape(np.asarray(images_2), [len(images_2), pixels])
     images_3 = np.reshape(np.asarray(images_3), [len(images_3), pixels])
     images_4 = np.reshape(np.asarray(images_4), [len(images_4), pixels])

     start = min(np.amin(images_1), np.amin(images_2), np.amin(images_3), np.amin(images_4))
     end = max(np.amax(images_1), np.amax(images_2), np.amax(images_3), np.amax(images_4))
     kl_divergence = np.zeros([4, 4, pixels])

     print start, end
     time_b = time.time()
     for i in range(pixels):
         images_pdf = []
         images_pdf.append(np.histogram(images_1[:,i:i+1], bins=100, range=(start, end), density=False)[0].astype(np.float32))
         images_pdf.append(np.histogram(images_2[:,i:i+1], bins=100, range=(start, end), density=False)[0].astype(np.float32))
         images_pdf.append(np.histogram(images_3[:,i:i+1], bins=100, range=(start, end), density=False)[0].astype(np.float32))
         images_pdf.append(np.histogram(images_4[:,i:i+1], bins=100, range=(start, end), density=False)[0].astype(np.float32))

         images_pdf[0][images_pdf[0] ==0] = 1e-7
         images_pdf[1][images_pdf[1] ==0] = 1e-7
         images_pdf[2][images_pdf[2] ==0] = 1e-7
         images_pdf[3][images_pdf[3] ==0] = 1e-7
         for k in range(4):
             for j in range(4):
                 p = images_pdf[k]
                 q = images_pdf[j]

                 kl_divergence[k][j][i] = scipy.stats.entropy(p, q)
     print time.time() - time_b

     np.save(open('/home/sukanya/Documents/'+intype+'/train_test_trim.npy', 'w+'), kl_divergence, allow_pickle=True)

def print_kl_divergence(intype):
     kl = np.load('/home/sukanya/Documents/'+intype+'/train_test_trim.npy', 'r')
     x,y,z = kl.shape
     kl_flat = np.reshape(kl, [x*y*z])
     index = largest_indices(kl_flat, 10000)
     print index
     
     plt.plot(kl_flat[index[0]])
     plt.show()

def get_orig_data_and_save(intype='fdg'):
    if intype == 'fdg':
        filename = fdg_filename
    if intype == 'av45':
        filename = av45_filename
    if intype == 'mri':
        filename = mri_filename

        # Matlab v7.3 format, hence the difference
        images = hdf5storage.loadmat(filename + 'split1.mat')['images'][0]
        for i in range(1,6):
            images = np.concatenate((images, hdf5storage.loadmat(filename + 'split'+str(i)+'.mat')['images'][0]), axis=0)

        image_label_mapping = scipy.io.loadmat(filename + 'details.mat')['foundinds'][0]
        num_images = len(image_label_mapping)
        height, width, depth = images[0].shape
        print images.shape, num_images
        image_new = np.zeros(shape=[num_images, height, width, depth], dtype=float)

        for i in range(num_images):
            image_new[i:i + 1, :, :, :] = images[image_label_mapping[i] - 1]

        np.save(open('/home/sukanya/Documents/' + intype + '/train_test.npy', 'w+'), image_new, allow_pickle=True)
        
    else:

        images = np.concatenate((scipy.io.loadmat(filename+'split1.mat')['images'][0],
                                np.concatenate((scipy.io.loadmat(filename+'split2.mat')['images'][0],
                                               scipy.io.loadmat(filename+'split3.mat')['images'][0]
                                                ))
                                ))
        image_label_mapping = scipy.io.loadmat(filename+'details.mat')['foundinds'][0]
        num_images = len(image_label_mapping)
        height, width, depth = images[0].shape
        print images.shape
        image_new = np.zeros(shape=[num_images, height, width, depth], dtype=float)

        for i in range(num_images):
            image_new[i:i+1,:,:,:] = images[image_label_mapping[i]-1]

        np.save(open('/home/sukanya/Documents/'+intype+'/train_test.npy', 'w+'),image_new, allow_pickle=True)

    labels = scipy.io.loadmat(filename + 'covariates.mat')['dxbl'][0].reshape((-1))
    labels[labels == 0] = np.random.randint(1, 5,
                                            size=len(labels[labels == 0]))  # Distributing 0's randomly to add noise

    np.save(open('/home/sukanya/Documents/'+intype+'/train_test_labels.npy', 'w+'),labels, allow_pickle=True)


def get_kl_divergence_data(intype):

    images = np.load('/home/sukanya/Documents/'+intype+'/train_test.npy', 'r')
    num_images, height, width, depth = images.shape
    print height, width, depth

    # Removing all NANs
    images_flattened = np.reshape(images, [num_images, height*width*depth])
    #images_mean = np.mean(images_flattened, axis=0)

    #isNan = np.isnan(images_flattened)
    #print np.isnan(images_flattened)
    print images_flattened.shape
    images_nan = images_flattened[:, ~np.any(np.isnan(images_flattened), axis=0)]
    print images_nan.shape
    
    labels = np.load('/home/sukanya/Documents/'+intype+'/train_test_labels.npy', 'r')

    get_kl_divergence(images_nan, labels, intype)
    

def get_data(one_hot_encoding=True, intype='fdg',kl_type='median', num_classes=2,trim=True, is_flat=True):
    # kl_type = None means just take the largest across all combinations, and get rid of repetitions
    # kl_type = 'mean' means take the mean of all the values across all combinations and then take the top n
    # kl_type = 'median' same as above but median instead of mean

    if intype == 'fdg':
        filename = fdg_filename
    if intype == 'av45':
        filename = av45_filename
    if intype == 'mri':
        filename = mri_filename

    images = np.load('/home/sukanya/Documents/'+intype+'/train_test.npy', 'r')
    num_images, height, width, depth = images.shape

    if is_flat:
        images_flattened = np.reshape(images, [num_images, height*width*depth])
        images = images_flattened

    if trim:
        kl_divergence = np.load('/home/sukanya/Documents/'+intype+'/train_test_trim.npy', 'r')

        if num_classes == 2:
            # Take only the first and last classes
            kl_divergence = kl_divergence[0:1,3:4,:] + kl_divergence[3:4,0:1,:]

        if kl_type == 'mean':
            kl_divergence_mean = np.reshape(kl_divergence, [kl_divergence.shape[0]*kl_divergence.shape[1], kl_divergence.shape[2]])
            kl_divergence_mean = np.mean(kl_divergence_mean, axis=0)

            indices = largest_indices(kl_divergence_mean, 80000)
            unique_indices = np.unique(indices[0])

        if kl_type == 'median':
            kl_divergence_median = np.reshape(kl_divergence,
                                       [kl_divergence.shape[0] * kl_divergence.shape[1], kl_divergence.shape[2]])
            kl_divergence_median = np.median(kl_divergence_median, axis=0)
            indices = largest_indices(kl_divergence_median, 60000)
            unique_indices = np.unique(indices[0])

        else:
            indices = largest_indices(kl_divergence, 120000)
            unique_indices = np.unique(indices[2])

            #print len(np.intersect1d(unique_indices, unique_indices_m)), len(np.intersect1d(unique_indices, unique_indices_mm)),
            #print len(np.intersect1d(unique_indices_mm, unique_indices_m))
        images_flattened_filtered = np.take(images, unique_indices, axis=1)
        images = images_flattened_filtered

        print images_flattened_filtered.shape

    labels = np.load('/home/sukanya/Documents/' + intype + '/train_test_labels.npy')
    labels[labels == 1] = 0
    labels[labels == 2] = 1
    labels[labels == 3] = 2
    labels[labels == 4] = 3
    counts = np.bincount(labels)

    count = [0.0]*4
    count[0] = 1.0*counts[0] / num_images
    count[1] = 1.0*counts[1] / num_images
    count[2] = 1.0*counts[2] / num_images
    count[3] = 1.0*counts[3] / num_images

     # TODO: Change this to cross validation
    num_test = 100
    labels = np.reshape(labels, [labels.shape[0], 1])

    images_labels = np.concatenate((images, labels), axis=1)

    test_set, train_set = np.split(images_labels[images_labels[:,images.shape[1]]==0].copy(), \
                                   [int(num_test*count[0]), ], axis=0)
    if num_classes == 2:
        list_start = 3
    else:
        list_start = 1

    for i in range(list_start,4):
        test_set_next, train_set_next = np.split(images_labels[images_labels[:,images.shape[1]]==i].copy(), \
                                           [int(num_test*count[i]), ], axis=0)
        test_set = np.concatenate((test_set, test_set_next), axis=0)
        train_set = np.concatenate((train_set, train_set_next), axis=0)

    train_set_images, train_set_labels = train_set[:, 0:images.shape[1]], np.reshape(
        train_set[:, images.shape[1]:images.shape[1] + 1], train_set.shape[0]).astype(int)

    test_set_images, test_set_labels = test_set[:, 0:images.shape[1]], np.reshape(
        test_set[:, images.shape[1]:images.shape[1] + 1], test_set.shape[0]).astype(int)

    print len(test_set), len(train_set), np.bincount(test_set_labels), np.bincount(train_set_labels)
    return mrbi_input.DataSet(train_set_images, mrbi_input.dense_to_one_hot(train_set_labels, 4), channels=True,
                              reshape=not is_flat), \
           mrbi_input.DataSet(test_set_images, mrbi_input.dense_to_one_hot(test_set_labels, 4), channels=True,
                              reshape=not is_flat)

    #print np.unique(test_set[:,:-1]), np.bincount(test_set[:,:-1])
    #num_train = len(labels) - num_test
    #return mrbi_input.DataSet(images[0:num_train], mrbi_input.dense_to_one_hot(labels[0:num_train], 4), channels=True, reshape=not is_flat),\
     #       mrbi_input.DataSet(images[num_train:num_train + 100], mrbi_input.dense_to_one_hot(labels[num_train:num_train + 100], 4), channels=True, reshape=not is_flat)


'''
get_data(kl_type='mean')
#print_kl_divergence('fdg')

print 'fdg'
get_orig_data_and_save(intype='av45')
get_kl_divergence_data(intype='av45')
#get_data(intype='fdg')

print 'av45'
get_data(intype='av45')
'''
#print 'mri'
#get_orig_data_and_save(intype='mri')
#get_kl_divergence_data(intype='mri')
#get_data(intype='mri')

