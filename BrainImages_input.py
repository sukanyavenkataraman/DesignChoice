# MRI, FDG and AV45 PET Scans
import numpy as np, scipy.io, h5py, scipy.stats, mrbi_input
from matplotlib import pyplot as plt
import time
import hdf5storage
from sklearn.model_selection import StratifiedShuffleSplit
from skimage.util import view_as_blocks, view_as_windows

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

     np.save(open('/home/sukanya/Documents/'+intype+'/train_test_trim_gtzero.npy', 'w+'), kl_divergence, allow_pickle=True)

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
    images_nan = images_flattened[:, ~np.any(np.isnan(images_flattened), axis=0)]
    images_nonzero = images_nan[:, ~(np.any(images_nan < 0, axis=0))]

    print images_nan.shape, images_nonzero.shape, np.where(images_nonzero < 0)[0]
    labels = np.load('/home/sukanya/Documents/'+intype+'/train_test_labels.npy', 'r')

    get_kl_divergence(images_nonzero, labels, intype)
    
def patch_extract_3D(input,patch_shape,xstep=1,ystep=1,zstep=1):
    patches_3D = np.lib.stride_tricks.as_strided(input, ((input.shape[0] - patch_shape[0] + 1) / xstep, (input.shape[1] - patch_shape[1] + 1) / ystep,
                                                  (input.shape[2] - patch_shape[2] + 1) / zstep, patch_shape[0], patch_shape[1], patch_shape[2]),
                                                  (input.strides[0] * xstep, input.strides[1] * ystep,input.strides[2] * zstep, input.strides[0], input.strides[1],input.strides[2]))
    patches_3D= patches_3D.reshape(patches_3D.shape[0]*patches_3D.shape[1]*patches_3D.shape[2], patch_shape[0],patch_shape[1],patch_shape[2])
    return patches_3D

def get_data(one_hot_encoding=True, intype='fdg', use_resnet=False, use_vgg=False, no_vgg_layers=0, kl_type='median', num_classes=4, k_fold=1, trim=True, is_flat=True, for_pretrain=False):
    # kl_type = None means just take the largest across all combinations, and get rid of repetitions
    # kl_type = 'mean' means take the mean of all the values across all combinations and then take the top n
    # kl_type = 'median' same as above but median instead of mean

    '''
    train_images = np.load('/home/sukanya/Documents/'+intype+'/train_test.npy', 'r')
    train_images_labels = np.load('/home/sukanya/Documents/'+intype+'/train_test_labels.npy', 'r')
    train_images = np.nan_to_num(train_images, copy=True)
    shuffleSplit = StratifiedShuffleSplit(n_splits=k_fold, test_size=0.15, random_state=44)
    for train, test in shuffleSplit.split(X=train_images, y=train_images_labels):
        train_set_images, train_set_labels = np.take(train_images, train, axis=0), np.take(train_images_labels, train, axis=0)
        test_set_images, test_set_labels = np.take(train_images, test, axis=0), np.take(train_images_labels, test, axis=0)

    np.save(open('/home/sukanya/Documents/' + intype + '/train_images_nonan.npy', 'w+'), train_set_images,
            allow_pickle=True)
    np.save(open('/home/sukanya/Documents/' + intype + '/train_labels_nonan.npy', 'w+'), train_set_labels,
            allow_pickle=True)
    np.save(open('/home/sukanya/Documents/' + intype + '/test_images_nonan.npy', 'w+'), test_set_images,
            allow_pickle=True)
    np.save(open('/home/sukanya/Documents/' + intype + '/test_labels_nonan.npy', 'w+'), test_set_labels,
            allow_pickle=True)
    return

    '''

    if use_vgg:
        train_images = np.load('/home/sukanya/Documents/' + intype + '/train_vgg_'+str(no_vgg_layers)+'.npy', 'r')
        test_images = np.load('/home/sukanya/Documents/' + intype + '/test_vgg_'+str(no_vgg_layers)+'.npy', 'r')
    elif use_resnet:
        train_images = np.load(intype+'_window_densenet_train.npz')['features']
        test_images = np.load(intype+'_window_densenet_test.npz')['features']
    else:
        train_images = np.load('/home/sukanya/Documents/' + intype + '/train_images_nonan.npy', 'r')
        test_images = np.load('/home/sukanya/Documents/' + intype + '/test_images_nonan.npy', 'r')

    train_labels = np.load('/home/sukanya/Documents/' + intype + '/train_labels_nonan.npy')
    test_labels = np.load('/home/sukanya/Documents/' + intype + '/test_labels_nonan.npy')

    num_images, height, width, depth = train_images.shape
    print train_images.shape

    '''
    pltimage = np.copy(train_images[:,:,:,])
    #plt.imshow(pltimage)
    #plt.show()
    flat = np.reshape(pltimage, [-1,height*width*depth])
    print len(flat[flat < 0]), np.where(flat < 0)[0]
    flat[flat < 0] = 1
    #flat = np.reshape(flat, [height, width])
    #print pltimage.shape, flat.shape
    #plt.imshow(flat)
    #plt.show()
    return
    '''

    if is_flat:
        train_images = np.reshape(train_images, [num_images, height*width*depth])
        test_images =  np.reshape(test_images, [-1, height*width*depth])
        train_images = train_images[:, ~(np.any(train_images < 0, axis=0))]
        '''
        print np.any(train_images < 0, axis=0)
        print len(np.where(train_images < 0)[0])
        train_images = train_images[~(np.where(train_images < 0)[0])]
        print train_images.shape
        '''
        test_images = test_images[:, ~(np.any(test_images < 0, axis=0))]
        print 'Is flat', train_images.shape, test_images.shape

    else:
        '''
        images = np.repeat(train_images[:,:,:,:,np.newaxis], 3, axis=4)
        images = np.reshape(images, newshape=(images.shape[0], images.shape[1], images.shape))
        test_images = np.repeat(test_images[:,:,:,:,np.newaxis], 3, axis=4)
        print images.shape, test_images.shape
        return images, test_images
        '''
        images = np.pad(train_images, pad_width=((0,0), (0, 0), (0,0), (1,1)), mode='edge')
        print images.shape
        image_patch = view_as_windows(images, window_shape=(1,79,95,3), step=(1,79,95,3)).squeeze()
        images_test = np.pad(test_images, pad_width=((0,0), (0, 0), (0,0), (1,1)), mode='edge')
        image_test_patch = view_as_windows(images_test, window_shape=(1, 79, 95, 3), step=(1, 79, 95, 3)).squeeze()

        print image_patch.shape, image_test_patch.shape
        image_patch = np.reshape(image_patch, newshape=(
            image_patch.shape[0]*image_patch.shape[1], image_patch.shape[2], image_patch.shape[3], image_patch.shape[4]))

        image_test_patch = np.reshape(image_test_patch, newshape=(
            image_test_patch.shape[0] * image_test_patch.shape[1], image_test_patch.shape[2], image_test_patch.shape[3],
            image_test_patch.shape[4]))

        print image_patch.shape, image_test_patch.shape
        return image_patch, image_test_patch

    if trim and not use_vgg and not use_resnet:
        kl_divergence = np.load('/home/sukanya/Documents/'+intype+'/train_test_trim_gtzero.npy', 'r')

        if num_classes == 2:
            # Take only the first and last classes
            kl_divergence = kl_divergence[0:1,3:4,:] + kl_divergence[3:4,0:1,:]

        if kl_type == 'mean':
            kl_divergence_mean = np.reshape(kl_divergence, [kl_divergence.shape[0]*kl_divergence.shape[1], kl_divergence.shape[2]])
            kl_divergence_mean = np.mean(kl_divergence_mean, axis=0)

            indices = largest_indices(kl_divergence_mean, 67500)
            unique_indices = np.unique(indices[0])

        elif kl_type == 'median':
            kl_divergence_median = np.reshape(kl_divergence,
                                       [kl_divergence.shape[0] * kl_divergence.shape[1], kl_divergence.shape[2]])
            kl_divergence_median = np.median(kl_divergence_median, axis=0)
            indices = largest_indices(kl_divergence_median, 67500)
            unique_indices = np.unique(indices[0])

        else:
            indices = largest_indices(kl_divergence, 120000)
            unique_indices = np.unique(indices[2])

        images = np.take(train_images, unique_indices, axis=1)
        test_images = np.take(test_images, unique_indices, axis=1)

    if use_vgg or use_resnet or not trim:
        images = train_images

    train_labels[train_labels == 1] = 0
    train_labels[train_labels == 2] = 1
    train_labels[train_labels == 3] = 2
    train_labels[train_labels == 4] = 3

    test_labels[test_labels == 1] = 0
    test_labels[test_labels == 2] = 1
    test_labels[test_labels == 3] = 2
    test_labels[test_labels == 4] = 3

    if use_resnet:
        train_labels = np.repeat(train_labels, 27)
        print train_labels.shape

        test_labels = np.repeat(test_labels, 27)
        print test_labels.shape

    print images.shape, test_images.shape

    if for_pretrain:
        return images, test_images

    shuffleSplit = StratifiedShuffleSplit(n_splits=k_fold, test_size=0.15, random_state=np.random.RandomState())

    for train, valid in shuffleSplit.split(X=images, y=train_labels):
        train_set_images, train_set_labels = np.take(images, train, axis=0), np.take(train_labels, train, axis=0)
        valid_set_images, valid_set_labels = np.take(images, valid, axis=0), np.take(train_labels, valid, axis=0)

    print len(train_set_labels), len(valid_set_images), len(test_labels)
    return mrbi_input.DataSet(train_set_images, mrbi_input.dense_to_one_hot(train_set_labels, 4), channels=True,
                              reshape=not is_flat), \
           mrbi_input.DataSet(valid_set_images, mrbi_input.dense_to_one_hot(valid_set_labels, 4), channels=True,
                              reshape=not is_flat), \
           mrbi_input.DataSet(test_images, mrbi_input.dense_to_one_hot(test_labels, 4), channels=True,
                              reshape=not is_flat)


    #print np.unique(test_set[:,:-1]), np.bincount(test_set[:,:-1])
    #num_train = len(labels) - num_test
    #return mrbi_input.DataSet(images[0:num_train], mrbi_input.dense_to_one_hot(labels[0:num_train], 4), channels=True, reshape=not is_flat),\
     #       mrbi_input.DataSet(images[num_train:num_train + 100], mrbi_input.dense_to_one_hot(labels[num_train:num_train + 100], 4), channels=True, reshape=not is_flat)


'''
get_data(kl_type='mean')
print_kl_divergence('fdg')

print 'fdg'
'''
#get_orig_data_and_save(intype='av45')
#get_kl_divergence_data(intype='mri')
#get_data(intype='fdg')
'''
print 'av45'
get_data(intype='av45')
'''
#print 'mri'
#get_orig_data_and_save(intype='mri')
#get_kl_divergence_data(intype='mri')
#get_data(intype='fdg')

