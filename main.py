import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
from dataset import trainingData

# Loading in the training and testing data
ds_train = (trainingData("Mask_Dataset"))
ds_test = (trainingData("Testing_Set"))

# Shape
# Training - (966, 1024, 1024, 3)
# Testing - (214, 1024, 1024, 3)

# Converting to float values from integers
ds_train = ds_train.astype('float32')
ds_test = ds_test.astype('float32')
print(ds_train.shape, ds_test.shape)

# Scaling values to be between 0 and 1 for performance
ds_train /= 255
ds_test /= 255

# Defining the model
model = keras.Sequential({
    keras.layers.Dense(512, input_dim=1024, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(1, activation='softmax')
})

# Defining the parameters of the model
model.compile(optimizer='adam',
              loss=keras.losses.sparse_categorical_crossentropy(),
              metrics=['accuracy'])

# Fitting the model
"""
Need to find a set of images with the mask on wrong to train the
model properly as well as to have propoer X and Y data.
"""
# model.fit(train_images)
