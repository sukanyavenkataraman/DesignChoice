
# coding: utf-8

# In[1]:


from keras import models
from keras import layers
from keras import metrics
from keras import optimizers
from keras import losses
from keras import activations
from keras import callbacks
from keras.utils import plot_model
from IPython.display import Image
from keras import Input
from keras import backend as K
import math


# In[2]:


model = models.Sequential()
model.add(layers.Conv3D(64, kernel_size=(5,7,7), input_shape=(79, 79, 95, 1), strides=(2,2,2), padding='valid', data_format='channels_last', activation=activations.relu))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling3D(data_format='channels_last', pool_size=(3,3,3),strides=(2,2,2), padding='same'))
model.add(layers.TimeDistributed(layers.Flatten()))
model.add(layers.GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='grui1'))
model.add(layers.Dense(512))
model.add(layers.Dense(4, activation='softmax'))
model.add(layers.Permute((2,1)))
model.add(layers.Lambda(lambda xin: K.sum(xin, axis=2)))
model.summary()


# In[6]:


model.compile(optimizer=optimizers.adam(lr=0.001, decay=0.9), loss=losses.categorical_crossentropy, metrics=[metrics.categorical_accuracy])


# In[ ]:


import BrainImages_input as input_data
all_data = input_data.get_data(intype='av45', trim=False, use_resnet=False, for_pretrain=False, is_flat=False, is_keras=True)

# Save best model
filepath="weights.best.hdf5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

hist = model.fit(all_data[0][0], all_data[0][1], validation_data=(all_data[1][0], all_data[1][1]), epochs=1, batch_size=16, callbacks=callbacks_list, verbose=0) # Add validation_split?

print (hist.history)

score = model.evaluate(all_data[1][0], all_data[1][1], batch_size=16)
print (score)

