# Importing tensorflow 
import tensorflow as tf

# Timing calculation
import time 

# Command line arguments
import sys 

# Reuters dataset
from keras.datasets import reuters

# keras API 
from keras import models, regularizers, layers

# Array math 
import numpy as np 

# Hidding the GPU
if len(sys.argv) > 1:
    if sys.argv[1] == 'cpu':
        tf.config.set_visible_devices([], 'GPU')

# Loading the mnist dataset 
(x_train, y_train), (x_test, y_test) = reuters.load_data(
    path="reuters.npz",
    num_words=10000
)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# Vectorize and Normalize train and test to tensors with 10k columns
x_train = vectorize_sequences(x_train)
x_test = vectorize_sequences(x_test)

print("x_train ", x_train.shape)
print("x_test ", x_test.shape)

# One hot encoding the labels
one_hot_train_labels = tf.keras.utils.to_categorical(y_train)
one_hot_test_labels = tf.keras.utils.to_categorical(y_test)

print("one_hot_train_labels ", one_hot_train_labels.shape)
print("one_hot_test_labels ", one_hot_test_labels.shape)

# Defining the model 
model = models.Sequential()
model.add(layers.Dense(1024, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(46, activation='softmax'))
model.summary()

# Compiling the model 
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

start = time.time()
model.fit(x_train, one_hot_train_labels, epochs=50, batch_size=1024, validation_data=(x_test, one_hot_test_labels))
training_time = time.time() - start 

# Printing the resulting time 
print(f"Training took: {training_time} seconds") 