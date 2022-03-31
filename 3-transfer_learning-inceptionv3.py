
import keras
import tensorflow as tf

conv_base = tf.keras.applications.InceptionV3(
                                            include_top=False,
                                            weights="imagenet",
                                            input_shape=(150, 150, 3)
                                        )

conv_base.summary()

from keras import models
from keras import layers
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

