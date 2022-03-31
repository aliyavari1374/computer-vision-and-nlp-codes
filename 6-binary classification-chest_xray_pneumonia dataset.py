
!pip install kaggle

!mkdir .kaggle

import json
token = {"username":"aliyavari1374","key":"725c54acb33fea292d42a7e127968e3d"}
with open('/content/.kaggle/kaggle.json', 'w') as file:
    json.dump(token, file)

!cp /content/.kaggle/kaggle.json ~/.kaggle/kaggle.json

!kaggle config set -n path -v{/content}

!chmod 600 /root/.kaggle/kaggle.json

!kaggle datasets list

#!kaggle datasets list -s sentiment
#!kaggle competitions download -c dogs-vs-cats -p /content
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
#!kaggle datasets download -d nih-chest-xrays/data         #42 GB Dataset

!cd /content/{/content}/datasets/paultimothymooney/chest-xray-pneumonia/

import shutil
source = '/content/{/content}/datasets/paultimothymooney/chest-xray-pneumonia/chest-xray-pneumonia.zip'
# Destination path  
destination = '/content'
# source to destination  
dest = shutil.move(source, destination)

#!kaggle datasets download -d kazanova/sentiment140 -p /content
!unzip \*.zip

import os, shutil
train_dir = "/content/chest_xray/train"
validation_dir = "/content/chest_xray/test"
train_normal_dir = "/content/chest_xray/train/NORMAL"
train_pneumonia_dir = "/content/chest_xray/train/PNEUMONIA"
validation_normal_dir = "/content/chest_xray/test/NORMAL"
validation_pneumonia_dir = "/content/chest_xray/test/PNEUMONIA"
#test_normal_dir = "/content/chest_xray/train/NORMAL"
#test_pneumonia_dir = "/content/chest_xray/train/NORMAL"

print('total training normal images:', len(os.listdir(train_normal_dir)))
print('total training pneumonia images:', len(os.listdir(train_pneumonia_dir)))
print('total validation normal images:', len(os.listdir(validation_normal_dir)))
print('total validation pneumonia images:', len(os.listdir(validation_pneumonia_dir)))
#print('total test cat images:', len(os.listdir(test_cats_dir)))
#print('total test dog images:', len(os.listdir(test_dogs_dir)))

# Example 1 
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
model.add(layers.Dropout(0.2))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# If Error is occured then this cell be run
pip install -U keras

# If Error is occured then this cell be run
!pip install tensorflow==1.14.0

# Example 1 
from keras import optimizers
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['acc'])
from keras.preprocessing.image import ImageDataGenerator 
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255) 
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150, 150), batch_size=128,class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=128,class_mode='binary')
history = model.fit_generator(train_generator,steps_per_epoch=40,epochs=30,validation_data=validation_generator,validation_steps=5)

model.save('cats_and_dogs_small_1.h5')

# Example 1 
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

###********************************************''' Data Augmentation (example 2-choilet)

model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150, 150),batch_size=128,class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(150, 150),batch_size=128,class_mode='binary')

history = model.fit_generator(train_generator,steps_per_epoch=40,epochs=50,validation_data=validation_generator,validation_steps=5)

model.save('cats_and_dogs_small_2.h5')

#(example 2)
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

###******************* Feature extraction Using Data Augmentation(example 4-choilet) - VGG16
from keras.applications import VGG16
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
conv_base.summary()

###******************* Feature extraction Using Data Augmentation(example 4-choilet) - Inception V3
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
conv_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
print("model structure: ", conv_base.summary())
#print("model weights: ", model.get_weights())

#Resume - (example 4-choilet)
from keras import models
from keras import layers
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# Resume - (example 4-choilet)
print('This is the number of trainable weights '
'before freezing the conv base:', len(model.trainable_weights))

conv_base.trainable = False
print('This is the number of trainable weights '  
'after freezing the conv base:', len(model.trainable_weights))

model.summary()

# Resume - (example 4-choilet)
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150, 150),batch_size=128,class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(150, 150),batch_size=128,class_mode='binary')

model.compile(loss='binary_crossentropy',optimizer=optimizers.SGD(),metrics=['acc'])

history = model.fit_generator(train_generator,steps_per_epoch=40,epochs=100,validation_data=validation_generator,validation_steps=5)

# Resume - (example 4-choilet)
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

# Fine tuning Pre-trained model - (example 5-choilet)
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
print('This is the number of trainable weights ', len(model.trainable_weights))

# Resume -Fine tuning Pre-trained model - (example 5-choilet)
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150, 150),batch_size=128,class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(150, 150),batch_size=128,class_mode='binary')

model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(lr=1e-5),metrics=['acc'])# lr=1e-5 => to prevent weight change
history = model.fit_generator(train_generator,steps_per_epoch=40,epochs=100,validation_data=validation_generator,validation_steps=5)

# Resume - (example 5-choilet)
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

train_dir = "/content/chest_xray/val"
test_generator = test_datagen.flow_from_directory(test_dir,target_size=(150, 150),batch_size=20,class_mode='binary')
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)