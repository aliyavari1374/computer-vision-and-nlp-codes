
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("3snort_all.csv")
train_data = dataset.iloc[:, :-1]
train_targets = dataset.iloc[:, -1]
train_targets

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
#train_targets = minmaxnormalization(train_targets,min(train_targets),max(train_targets),0,1)
train_data

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
train_data

train_data = (train_data-train_data.mean())/(train_data.std())
train_data

sample_size = train_data.shape[0] # number of samples in train set
time_steps  = train_data.shape[1] # number of features in train set
input_dimension = 1               # each feature is represented by 1 number

print(sample_size)
print(time_steps)

train_data_reshaped = train_data.values.reshape(sample_size,time_steps,input_dimension)
print("After reshape train data set shape:\n", train_data_reshaped.shape)
print("1 Sample shape:\n",train_data_reshaped[0].shape)
print("An example sample:\n", train_data_reshaped[0])

from keras import layers
from keras import models
from keras import optimizers
from tensorflow.keras import regularizers
def build_conv1D_model():
    n_timesteps = train_data_reshaped.shape[1] # features in train_data
    n_features  = train_data_reshaped.shape[2] #1 
    model = models.Sequential(name="model_conv1D")
    
    model.add(layers.Conv1D(filters=16, kernel_size=7, activation='relu',padding='valid', name="B1_Conv1D_1",input_shape=(n_timesteps,n_features)))
    model.add(layers.Conv1D(filters=16, kernel_size=7, activation='relu',padding='valid', name="B1_Conv1D_2"))
    model.add(layers.Dropout(0.1,name="Drop1"))
    model.add(layers.MaxPooling1D(pool_size=2, name="B1_MaxPooling1D"))

    model.add(layers.Conv1D(filters=32, kernel_size=7, activation='relu',padding='valid', name="B2_Conv1D_1"))
    model.add(layers.Conv1D(filters=32, kernel_size=7, activation='relu',padding='valid', name="B2_Conv1D_2"))
    model.add(layers.Dropout(0.1,name="Drop2"))
    model.add(layers.MaxPooling1D(pool_size=2, name="B2_MaxPooling1D"))


    model.add(layers.Flatten(name="Flatten"))
    model.add(layers.Dense(1024, activation='relu', name="Dense_1"))
    model.add(layers.Dropout(0.1,name="Drop3"))
    model.add(layers.Dense(n_features, name="Dense_2"))

    # method 1 Scheduling ExponentialDecay
    #lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,decay_steps=300,decay_rate=0.9)
    #optimizer = optimizers.Adam(learning_rate=lr_schedule)

    # method 2 Scheduling InverseTimeDecay
    #initial_learning_rate = 1.0
    #decay_steps = 1.0
    #decay_rate = 0.5
    #learning_rate_fn = optimizers.schedules.InverseTimeDecay(initial_learning_rate, decay_steps, decay_rate)

    # method 3 Scheduling PolynomialDecay
    starter_learning_rate = 0.001
    end_learning_rate = 0.0001
    decay_steps = 10000
    learning_rate_fn = optimizers.schedules.PolynomialDecay(starter_learning_rate,decay_steps,end_learning_rate,power=0.5)
    model.compile(loss='mse',optimizer=optimizers.Adam(learning_rate_fn),metrics=['mae'])  
    #model.compile(loss='mae',optimizer=optimizers.RMSprop(learning_rate_fn))  

    return model

model_conv1D = build_conv1D_model()
model_conv1D.summary()

# Store training stats
EPOCHS = 1000
history = model_conv1D.fit(train_data_reshaped, train_targets,batch_size=1, epochs=EPOCHS, validation_split=0.1, verbose=1)

from keras import layers
from keras import models
from keras import optimizers
def build_VGG16conv1D_model():
    n_timesteps = train_data_reshaped.shape[1] # features in train_data
    n_features  = train_data_reshaped.shape[2] #1 
    model = models.Sequential(name="model_conv1D")
    
    # Block 1
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu',padding='same', name="B1Conv1D_1",input_shape=(n_timesteps,n_features)))
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu',padding='same', name="B1Conv1D_2"))
    model.add(layers.MaxPooling1D(pool_size=2, name="B1MaxPooling1D"))

    # Block 2
    model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu',padding='same', name="B2Conv1D_1"))
    model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu',padding='same', name="B2Conv1D_2"))
    model.add(layers.MaxPooling1D(pool_size=2, name="B2MaxPooling1D"))

    # Block 3
    model.add(layers.Conv1D(filters=256, kernel_size=3, activation='relu',padding='same', name="B3Conv1D_1"))
    model.add(layers.Conv1D(filters=256, kernel_size=3, activation='relu',padding='same', name="B3Conv1D_2"))
    model.add(layers.Conv1D(filters=256, kernel_size=3, activation='relu',padding='same', name="B3Conv1D_3"))
    model.add(layers.MaxPooling1D(pool_size=2, name="B3MaxPooling1D"))
   
    # Block 4
    model.add(layers.Conv1D(filters=512, kernel_size=3, activation='relu',padding='same', name="B4Conv1D_1"))
    model.add(layers.Conv1D(filters=512, kernel_size=3, activation='relu',padding='same', name="B4Conv1D_2"))
    model.add(layers.Conv1D(filters=512, kernel_size=3, activation='relu',padding='same', name="B4Conv1D_3"))
    model.add(layers.MaxPooling1D(pool_size=2, name="B4MaxPooling1D"))

    # Block 5
    model.add(layers.Conv1D(filters=512, kernel_size=3, activation='relu',padding='same', name="B5Conv1D_1"))
    model.add(layers.Conv1D(filters=512, kernel_size=3, activation='relu',padding='same', name="B5Conv1D_2"))
    model.add(layers.Conv1D(filters=512, kernel_size=3, activation='relu',padding='same', name="B5Conv1D_3"))   
    model.add(layers.MaxPooling1D(pool_size=2, name="B5MaxPooling1D"))

    # Flatten
    model.add(layers.Flatten(name="Flatten_Layer"))

    # Dense layers
    model.add(layers.Dense(2048, activation='relu', name="Dense_1"))       
    model.add(layers.Dense(n_features, name="Dense_2"))


     # method 3 Scheduling PolynomialDecay
    starter_learning_rate = 0.001
    end_learning_rate = 0.0001
    decay_steps = 10000
    learning_rate_fn = optimizers.schedules.PolynomialDecay(starter_learning_rate,decay_steps,end_learning_rate,power=0.5)
    model.compile(loss='mse',optimizer=optimizers.Adam(learning_rate_fn),metrics=['mae'])    

    return model

model_VGG16conv1D = build_VGG16conv1D_model()
model_VGG16conv1D.summary()

# Store training stats
EPOCHS = 250
history = model_VGG16conv1D.fit(train_data_reshaped, train_targets,batch_size=1, epochs=EPOCHS, validation_split=0.2, verbose=1)

import matplotlib.pyplot as plt
def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(history.epoch, np.array(history.history['mae']), 
           label='Train')
  plt.plot(history.epoch, np.array(history.history['val_mae']),
           label = 'Val')
  plt.legend()
  plt.ylim([0,max(history.history['val_mae'])])

def plot_prediction(test_labels, test_predictions):
  plt.figure()
  plt.scatter(test_labels, test_predictions)
  plt.xlabel('True Values [1000$]')
  plt.ylabel('Predictions [1000$]')
  plt.axis('equal')
  plt.xlim(plt.xlim())
  plt.ylim(plt.ylim())
  _ = plt.plot([-100, 100],[-100,100])
  plt.savefig("squares.png",dpi=400) 

  plt.figure()
  error = test_predictions - test_labels
  plt.hist(error, bins = 50)
  plt.xlabel("Prediction Error [1000$]")
  _ = plt.ylabel("Count")

plot_history(history)

import numpy as np
k = 10
num_val_samples = len(train_data_reshaped) // k
num_epochs = 100
all_mae_histories = []
for i in range(k):
    print('processing fold #', i+1)
    val_data = train_data_reshaped[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data_reshaped[:i * num_val_samples],train_data_reshaped[(i + 1) * num_val_samples:]],axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],train_targets[(i + 1) * num_val_samples:]],axis=0)
    model_conv1D = build_conv1D_model()
    #model_VGG16conv1D = build_VGG16conv1D_model()
    history = model_conv1D.fit(partial_train_data, partial_train_targets,validation_data=(val_data, val_targets),epochs=num_epochs, batch_size=1, verbose=1)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()