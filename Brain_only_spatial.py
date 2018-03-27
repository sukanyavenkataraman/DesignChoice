
# coding: utf-8

# In[1]:


from keras import models
from keras import layers
from keras import metrics
from keras import optimizers
from keras import losses
from keras import activations
from keras import callbacks
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.utils import plot_model
from IPython.display import Image
from keras import Input
from keras import backend as K
import math


# In[2]:


import BrainImages_input as input_data
import numpy as np

all_data = input_data.get_data(intype='av45', trim=False, use_resnet=False, for_pretrain=False, is_flat=False, is_keras=True)

num, depth, width, height, channels = all_data[0][0].shape

all_data[0][0] = np.reshape(all_data[0][0], newshape=(num*depth, width, height, channels))
all_data[0][0] = np.pad(all_data[0][0], pad_width=((0,0), (60,61), (52,53), (0,0)), mode='edge')
all_data[0][0] = np.repeat(all_data[0][0], 3, axis=3)
all_data[0][1] = np.repeat(all_data[0][1], depth, axis=0)

all_data[1][0] = np.reshape(all_data[1][0], newshape=(all_data[1][0].shape[0]*depth, width, height, channels))
all_data[1][0] = np.pad(all_data[1][0], pad_width=((0,0), (60,61), (52,53), (0,0)), mode='edge')
all_data[1][0] = np.repeat(all_data[1][0], 3, axis=3)
all_data[1][1] = np.repeat(all_data[1][1], depth, axis=0)

all_data[2][0] = np.reshape(all_data[2][0], newshape=(all_data[2][0].shape[0]*depth, width, height, channels))
all_data[2][0] = np.pad(all_data[2][0], pad_width=((0,0), (60,61), (52,53), (0,0)), mode='edge')
all_data[2][0] = np.repeat(all_data[2][0], 3, axis=3)
all_data[2][1] = np.repeat(all_data[2][1], depth, axis=0)


# In[3]:


model = models.Sequential()
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(200, 200, 3), pooling='max')
model.add(base_model)
model.add(layers.Dense(512))
model.add(layers.Dense(4, activation='softmax'))
model.summary()

for layer in base_model.layers:
    layer.trainable = False


# In[4]:


model.compile(optimizer=optimizers.adam(lr=0.001, decay=0.9), loss=losses.categorical_crossentropy, metrics=[metrics.categorical_accuracy])


# In[ ]:



# Save best model
filepath="weights_resnet.best.hdf5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

print ('going to train')

hist = model.fit(all_data[0][0], all_data[0][1], validation_data=(all_data[1][0], all_data[1][1]), epochs=1, batch_size=64, callbacks=callbacks_list, verbose=1) # Add validation_split?

print (hist.history)
print('going to test')
score = model.evaluate(all_data[2][0], all_data[2][1], batch_size=64)
print (score)

