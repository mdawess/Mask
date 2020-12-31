import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
from dataset import trainingData, trainingLabels

# Loading in the training and testing data (x and y var for each)
ds_train = trainingData("training_set")  # x variable
ds_test = trainingData("testing_set")
training_labels = trainingLabels("training_set")  # y variable
testing_labels = trainingLabels("testing_set")

# Shape
# Training - (966, 1024, 1024, 3)
# Testing - (214, 1024, 1024, 3)

# Converting to float values from integers
ds_train = ds_train.astype('float32')
ds_test = ds_test.astype('float32')
# print(ds_train.shape, ds_test.shape)

# Scaling values to be between 0 and 1 for performance
ds_train /= 255
ds_test /= 255

# Defining the model
model = keras.models.Sequential()
model.add(keras.layers.Dense(512, input_dim=1024, activation='relu',))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(2, activation='sigmoid'))

# # Defining the parameters of the model
model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['accuracy'])

# Fitting the model
"""
Need to find a set of images with the mask on wrong to train the
model properly as well as to have propoer X and Y data.
"""
# model.fit(train_images)
