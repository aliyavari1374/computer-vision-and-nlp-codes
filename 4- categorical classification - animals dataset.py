from sklearn.neural_network import MLPClassifier
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')

"""
************************1-Download Data****************************
"""
import os, shutil
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.utils import plot_model

dataset_dir = "/content/drive/My Drive/Colab Notebooks/datasets/animals/"

train_dir = dataset_dir + "train/"
validation_dir = dataset_dir + "validation/"
test_dir = dataset_dir + "test/"


train_cats_dir = train_dir + "cats"
train_dogs_dir = train_dir + "dogs"
train_panda_dir = train_dir + "panda"

validation_cats_dir = validation_dir + "cats"
validation_dogs_dir = validation_dir + "dogs"
validation_panda_dir = validation_dir + "panda"

test_cats_dir =  test_dir + "cats"
test_dogs_dir =  test_dir + "dogs"
test_panda_dir =  test_dir + "panda"

print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total training dog images:', len(os.listdir(train_panda_dir)))

print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total validation dog images:', len(os.listdir(validation_panda_dir)))

print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir))) 
print('total test dog images:', len(os.listdir(test_panda_dir)))

"""
************************2-Building your network****************************
"""

from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

model.summary()

plot_model(model, to_file="mynet.pdf", show_shapes=True)

from keras import optimizers
model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=1e-4),metrics=['acc'])

"""
************************3-Generate And Feed Data to NN****************************
"""

from keras.preprocessing.image import ImageDataGenerator

import numpy as np

train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='categorical')

# For Test Batch Performance
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    for j in range(0,len(data_batch)-15):  
        plt.figure(j)        
        imgplot = plt.imshow(image.array_to_img(data_batch[j]))     
    
    break
plt.show()

# Fitting Data to Network
history = model.fit(train_generator,steps_per_epoch=75,
                              epochs=30,validation_data=validation_generator,
                              validation_steps=37)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir,target_size=(150, 150),batch_size=20,class_mode='categorical')
test_loss, test_acc = model.evaluate(test_generator, steps=37)
print('test acc:', test_acc)

# INCEPTION
import keras
import tensorflow as tf
conv_base = tf.keras.applications.InceptionV3(
                                            include_top=False,
                                            weights="imagenet",
                                            input_shape=(150, 150, 3)
                                        )
#conv_base.summary()

from keras import models
from keras import layers
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.summary()

print('This is the number of trainable weights '
'before freezing the conv base:', len(model.trainable_weights))

conv_base.trainable = False
print('This is the number of trainable weights '  
'after freezing the conv base:', len(model.trainable_weights))

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150, 150),batch_size=128,class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_dir,target_size=(150, 150),batch_size=128,class_mode='categorical')

model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=1e-4),metrics=['acc'])

history = model.fit(train_generator,steps_per_epoch=11,epochs=50,validation_data=validation_generator,validation_steps=5)

model.save(dataset_dir + "inceptionv3_Animals")

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir,target_size=(150, 150),batch_size=128,class_mode='categorical')
test_loss, test_acc = model.evaluate(test_generator, steps=5)
print('test acc:', test_acc)