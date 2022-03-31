

import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("bestsofar_dataset10.csv")
train_data = dataset.iloc[:, :-1]
train_targets = dataset.iloc[:, -1]
dataset

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)

#train_targets = minmaxnormalization(train_targets,min(train_targets),max(train_targets),0,1)
train_data

train_data = (train_data-train_data.mean())/(train_data.std())
train_data

from keras import models
from keras import layers
from tensorflow.keras import regularizers
from keras import optimizers
def build_model():
    model = models.Sequential(name="Model1")
    model.add(layers.Dense(512,activation='relu',input_shape=(train_data.shape[1],)))    
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dropout(0.2)) 
    model.add(layers.Dense(512,activation='relu')) 
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(1))


    starter_learning_rate = 0.1
    end_learning_rate = 0.0001
    decay_steps = 1000
    learning_rate_fn = optimizers.schedules.PolynomialDecay(starter_learning_rate,decay_steps,end_learning_rate,power=0.5)    
    model.compile(loss='mse',optimizer=optimizers.Adam(learning_rate_fn),metrics=['mae'])  
    return model

model = build_model()
model = model.summary()

# Store training stats
EPOCHS = 300
model = build_model()
model.fit(train_data, train_targets,epochs=EPOCHS, batch_size=1,validation_split=0.2, verbose=1)

import numpy as np
k = 10
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print('processing fold #', i+1)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],train_targets[(i + 1) * num_val_samples:]],axis=0)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,epochs=num_epochs, batch_size=1, verbose=1)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

import numpy as np
k = 10
num_val_samples = len(train_data) // k
num_epochs = 250
all_mae_histories = []
for i in range(k):
    print('processing fold #', i+1)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],train_data[(i + 1) * num_val_samples:]],axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],train_targets[(i + 1) * num_val_samples:]],axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,validation_data=(val_data, val_targets),epochs=num_epochs, batch_size=1, verbose=1)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()