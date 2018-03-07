import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
'''
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")
ax.quiver(0,0,0,3,0,0, color='red')
ax.quiver(0,0,0,1,0,0, color='blue')
ax.quiver(0,0,0,0,0.6,0.8, color='green')
ax.quiver(0,0,0,1,3,4, color='orange')

ax.set_xlim([-1,5])
ax.set_ylim([-1,5])
ax.set_zlim([-1,5])

plt.show()
'''

import numpy

h = [[33, 4], [55,5]]
print(numpy.asarray(h))
print(numpy.asarray(h).transpose())

k = [3, 2]
print(numpy.asarray(k)[numpy.newaxis])
print(numpy.asarray(k)[numpy.newaxis].transpose())

#Sorting based on loss and getting hls

hls = numpy.asarray(hidden_layer_sizes)
            loss_np = numpy.asarray(loss)[numpy.newaxis].transpose()

            hls_loss = numpy.append(hls, loss_np, axis=1)
            hls_loss = hls_loss[hls_loss[:,design_choice.network_depth-1].argsort()]

            loss_sorted = hls_loss[:,design_choice.network_depth-1]
            hls_sorted = hls_loss[:,0:design_choice.network_depth-1]

            max_loss = loss_sorted[len(loss_sorted) - 1]
            min_loss = loss_sorted[0]
            max_loss_hls = hls_sorted[len(loss_sorted) - 1]
            min_loss_hls = hls_sorted[0]

            max_loss_phis = [phi_hidden_layer[loss.index(max_loss)], phi_op_minus1[loss.index(max_loss)], phi_op[loss.index(max_loss)]]
            min_loss_phis = [phi_hidden_layer[loss.index(min_loss)], phi_op_minus1[loss.index(min_loss)], phi_op[loss.index(min_loss)]]


# Testing if this actually works

   randnum = random.uniform(0, 1)
   print(randnum)

   value = float(min(hls_cdf, key=lambda x:abs(x-randnum)))
   index = cdf_phis.index(value)

   print(index)
   print(hls_cdf.get(value))

''''
            x = layers.conv2d_block("CONV2D", x, 64, 1, 1, p="SAME", data_format="NCHW", bias=True, bn=False,
                                    activation_fn=tf.nn.relu)
            print x.get_shape
            in_shape = x.get_shape()[-1]
            print in_shape
            '''

import h5py
filename = '/home/sukanya/Downloads/vgg16_weights.h5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())

import numpy as np



weights = np.load('/home/sukanya/Downloads/vgg16_weights.npz')
keys = sorted(weights.keys())
print keys
for i, k in enumerate(keys):
    print i, k, np.shape(weights[k])


