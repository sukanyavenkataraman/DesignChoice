# MRI, FDG and AV45 PET Scans
import numpy as np, scipy.io, h5py, mrbi_input
from matplotlib import pyplot as plt

base_dir = '/media/vamsi/datadrive2tb/'
fdg_filename = base_dir+'FDG_3visits/tp1_fdgs/fdgs_tp1_'
av45_filename = base_dir+'AV45_3visits/tp1_av45s/av45s_tp1_'
mri_filename = base_dir+'MRIs_3visits/tp1_mris/mris_tp1_'

def get_data(one_hot_encoding=True, intype='fdg'):

    if intype == 'fdg':
        filename = fdg_filename
    if intype == 'av45':
        filename = av45_filename
    if intype == 'mri':
        filename = mri_filename

        # Matlab v7.3 format
        images = h5py.File(filename+'split1.mat') # Read this properly

    else:

        images = np.concatenate((scipy.io.loadmat(filename+'split1.mat')['images'][0],
                                np.concatenate((scipy.io.loadmat(filename+'split2.mat')['images'][0],
                                               scipy.io.loadmat(filename+'split3.mat')['images'][0]
                                                ))
                                ))
        '''

        images = scipy.io.loadmat(filename+'split1.mat')['images'][0]
        '''
        num_images = len(images)
        height, width, depth = images[0].shape
        print images.shape
        image_new = np.zeros(shape=[num_images, height, width, depth], dtype=float)

        for i in range(num_images):
            image_new[i:i+1,:,:,:] = images[i]
        '''
        to_plot = image_new.transpose(0,3,2,1)

        plt.imshow(to_plot[0][1])
        plt.show()

        plt.imshow(to_plot[0][10])
        plt.show()

        print num_images, height, width, depth, images.shape, image_new.shape
        '''
    labels = scipy.io.loadmat(filename+'covariates.mat')['dxbl'][0].reshape((-1))
    labels[labels == 0] = np.random.randint(1,5, size=len(labels[labels == 0])) # Distributing 0's randomly to add noise

    num_images = len(labels) # Since currently, labels < images

    labels_new = np.zeros(shape=[len(labels)])

    for i in range(num_images):
        labels_new[i] = labels[i]

    print labels_new.shape

    # TODO: Change this to cross validation
    num_test = 100
    return mrbi_input.DataSet(image_new[0:num_images-num_test], mrbi_input.dense_to_one_hot(labels[0:num_images-num_test], 4), channels=True),\
           mrbi_input.DataSet(image_new[num_images - num_test:num_images - num_test + 100], mrbi_input.dense_to_one_hot(labels[num_images - num_test:num_images - num_test + 100], 4), channels=True)

#print 'fdg'
#get_data(intype='fdg')
'''
print 'av45'
get_data(intype='av45')
print 'mri'
get_data(intype='mri')
'''