from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Input, AveragePooling2D
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.datasets import cifar10
import pickle
import tensorflow as tf
import keras.backend as K

h = 197
w = 197
ch = 3
batch_size=64
#tensor. will receive cifar10 images as input, gets passed to resize_images
img_placeholder = tf.placeholder("uint8", (None, 32, 32, 3))

#tensor. resized images. gets passed into Session() 
resize_op = tf.image.resize_images(img_placeholder, (h, w), method=0)

def gen(session, data, labels, batch_size):
    def _f():
        start = 0
        end = start + batch_size
        n = data.shape[0]
        while True:
            # run takes in a tensor/function and performs it.
            # almost always, that function will take a Tensor as input
            # when run is called, it takes a feed_dict param which translates
            # Tensors into actual data/integers/floats/etc
            # this is so you can write a network and only have to change the 
            # data being passed in one place instead of everywhere
            
            # X_batch is resized
            X_batch = session.run(resize_op, {img_placeholder: data[start:end]})
            # X_batch is normalized
            X_batch = preprocess_input(X_batch)
            y_batch = labels[start:end]
            start += batch_size
            end += batch_size
            if start >= n:
                start = 0
                end = batch_size
                print("Bottleneck predictions completed.")

            yield (X_batch, y_batch)

    return _f

def create_model_resnet():
    input_tensor = Input(shape=(h, w, ch))
    model = ResNet50(input_tensor=input_tensor, include_top=False)
    return model

(X_train, y_train), (_, _) = cifar10.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

with tf.Session() as sess:
    K.set_session(sess)
    K.set_learning_phase(1)

    model = create_model_resnet()

    train_gen = gen(sess, X_train, y_train, batch_size)
    bottleneck_features_train = model.predict_generator(train_gen(), 2000)
    data = {'features': bottleneck_features_train, 'labels': y_train[:2000]}
    pickle.dump(data, open('resnet_train_bottleneck.p', 'wb'))

    val_gen = gen(sess, X_val, y_val, batch_size)
    bottleneck_features_validation = model.predict_generator(val_gen(), 2000)
    data = {'features': bottleneck_features_validation, 'labels': y_val[:2000]}
    pickle.dump(data, open('resnet_validate_bottleneck.p', 'wb'))

X_train, y_train, X_val, y_val = load_bottleneck_data('resnet_train_bottleneck.p', 
                                                      'resnet_train_bottleneck.p')

nb_classes=10
input_shape = X_train.shape[1:]
inp = Input(shape=input_shape)
x = Flatten()(inp)
x = Dense(nb_classes, activation='softmax')(x)
model = Model(inp, x)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

with tf.Session() as sess:
    # fetch session so Keras API can work 
    K.set_session(sess)
    K.set_learning_phase(1)
    resnet_history =model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch_size,
                       validation_data=(X_val, y_val), shuffle=True, verbose=0)
    model.save_weights('resnet_bottleneck_weights.h5')


print(resnet_history.history)
