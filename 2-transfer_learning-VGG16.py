# Transfer Learning using VGG16 - a simple example

import tensorflow as tf


conv_base = tf.keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3)
)

conv_base.summary()

from keras import models
from keras import layers
def bulid_model():
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))
    return model

model = bulid_model()
model.build(input_shape=((None, 224, 224, 3)))
model.summary()

from keras import optimizers
model.compile(optimizer=optimizers.Adam(1e-3),loss='categorical_crossentropy',metrics=['acc'])

history = model.fit_generator(train_generator,steps_per_epoch=100,epochs=50)

