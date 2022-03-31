
from keras import layers
from keras import models
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist

# Load Data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print(train_images.shape)
print(test_labels.shape)

# Method 1 for building LeNet5 using API
from keras import Input, layers
from keras.models import Model
def build_model_api():
  input_tensor = Input(shape=(28,28,1),name="input")
  x = layers.Conv2D(6, 5, strides=1, activation='relu', input_shape=(28, 28,1), name="conv1")(input_tensor)
  x = layers.MaxPooling2D((2, 2), name="pool1")(x)
  x = layers.Conv2D(16, 5, strides=1, activation='relu', name="conv2")(x)
  x = layers.MaxPooling2D((2, 2), name="pool2")(x)
  x = layers.Flatten(name="flatten")(x)
  x = layers.Dense(120, activation='relu', name="dense1")(x)
  x = layers.Dense(84, activation='relu', name="dense2")(x)
  output_tensor = layers.Dense(10, activation='softmax', name="dense3")(x)
  model = Model(input_tensor, output_tensor)
  return model

model = build_model_api()
model.summary()

# Method 2 for building LeNet5 using Sequential
def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(6, 5, strides=1, activation='relu', input_shape=(28, 28,1), name="conv1"))
    model.add(layers.MaxPooling2D((2, 2), name="pool1"))
    model.add(layers.Conv2D(16, 5, strides=1, activation='relu', name="conv2"))
    model.add(layers.MaxPooling2D((2, 2), name="pool2"))
    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(120, activation='relu', name="dense1"))
    model.add(layers.Dense(84, activation='relu', name="dense2"))
    model.add(layers.Dense(10, activation='softmax', name="dense3"))
    return model
model = build_model()
model.summary()

# compiling the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# training
model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_split=0.2)

# evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

model.save('digit_recognition.h5')

# visualization
import matplotlib.pyplot as plt
image = test_images[0]
image.resize(28,28)

plt.imshow(image, cmap=plt.cm.binary)
plt.show()

import numpy as np
image = np.expand_dims(image, axis=0)
print(image.shape)

model.layers[0:]

from keras import models
layer_outputs = [layer.output for layer in model.layers[:4]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(image)
first_layer_activation = activations[2]
print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 13], cmap='viridis')

