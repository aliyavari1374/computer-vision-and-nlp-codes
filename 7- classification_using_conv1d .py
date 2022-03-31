
import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("z_datasetB - Copy.csv")
train_data = dataset.iloc[:, :-1]
train_targets = dataset.iloc[:, -1]
print(train_data.shape[1])
train_targets

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
#train_targets = minmaxnormalization(train_targets,min(train_targets),max(train_targets),0,1)
train_data

train_data[:,0]

sample_size = train_data.shape[0] # number of samples in train set
time_steps  = train_data.shape[1] # number of features in train set
input_dimension = 1               # each feature is represented by 1 number

print(sample_size)
print(time_steps)

train_data_reshaped = train_data.reshape(sample_size,time_steps,input_dimension)
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
    
    model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu',padding='valid', name="B1_Conv1D_1",input_shape=(n_timesteps,n_features)))
    model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu',padding='valid', name="B1_Conv1D_2"))
    model.add(layers.Dropout(0.1,name="Drop1"))
    model.add(layers.MaxPooling1D(pool_size=2, name="B1_MaxPooling1D"))

    model.add(layers.Conv1D(filters=256, kernel_size=3, activation='relu',padding='valid', name="B2_Conv1D_1"))
    model.add(layers.Conv1D(filters=256, kernel_size=3, activation='relu',padding='valid', name="B2_Conv1D_2"))
    model.add(layers.Dropout(0.1,name="Drop2"))
    model.add(layers.MaxPooling1D(pool_size=2, name="B2_MaxPooling1D"))


    model.add(layers.Flatten(name="Flatten"))
    model.add(layers.Dense(256, activation='relu', name="Dense_1"))
    model.add(layers.Dropout(0.1,name="Drop3"))
    model.add(layers.Dense(n_features,activation='sigmoid',name="Dense_2"))

    model.compile(optimizer=optimizers.Adam(1e-3),loss='binary_crossentropy',metrics=['acc'])
    return model

model_conv1D = build_conv1D_model()
model_conv1D.summary()

# Store training stats
EPOCHS = 50
history = model_conv1D.fit(train_data_reshaped, train_targets,batch_size=1, epochs=EPOCHS, validation_split=0.2, verbose=1)

import numpy as np
k = 5
num_val_samples = len(train_data_reshaped) // k
num_epochs = 25
all_val_acc_histories = []
all_acc_histories = []
for i in range(k):
    print('processing fold #', i+1)
    val_data = train_data_reshaped[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data_reshaped[:i * num_val_samples],train_data_reshaped[(i + 1) * num_val_samples:]],axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],train_targets[(i + 1) * num_val_samples:]],axis=0)
    model_conv1D = build_conv1D_model()
    
    history = model_conv1D.fit(partial_train_data, partial_train_targets,validation_data=(val_data, val_targets),epochs=num_epochs, batch_size=1, verbose=1)
    val_acc_history = history.history['val_acc']
    acc_history = history.history['acc']
    all_val_acc_histories.append(val_acc_history)
    all_acc_histories.append(acc_history)

average_vall_acc_history = [np.mean([x[i] for x in all_val_acc_histories]) for i in range(num_epochs)] 
average_acc_history = [np.mean([x[i] for x in all_acc_histories]) for i in range(num_epochs)]

import matplotlib.pyplot as plt
plt.plot(range(1, len(average_vall_acc_history) + 1), average_vall_acc_history)
plt.plot(range(1, len(average_acc_history) + 1), average_acc_history)
plt.xlabel('Epochs')
plt.ylabel('Validation ACC')
plt.show()