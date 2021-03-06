
from keras.datasets import imdb
from keras import preprocessing
max_features = 10000
maxlen = 100
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, Dropout
model = Sequential()
model.add(Embedding(max_features, 64, input_length=maxlen))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(x_train, y_train,
              epochs=10,
              batch_size=32,
              validation_split=0.2)

