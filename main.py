import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
from dataset import getData

# Loading in the training data
ds_train = getData("training_set")[0]  # x variable
training_labels = getData("training_set")[1]  # y variable


def formatVariable(x):
    """Formats the x variable to be of type float, reshapes
    the x variable to be greyscale (1 at end of shape) and scales the
    x variables to be between 0 and 1.
    """
    x = x.astype('float32').reshape((x.shape[0], x.shape[1], x.shape[2], 1))
    x /= 255

    return x


def buildModel(x, y):
    """Generates and fits the model using the inputted x and
    y variables. num_inputs is the number of items within the dataset
    being fed to the model. In this case of images, length and width
    refer to the dimensions of the image. Returns the Model.
    """
    x = formatVariable(x)
    # Defining the model
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten())

    # Add conv2D
    model.add(keras.layers.Dense(12, activation='relu',))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dropout(0.4))

    # Only need 1 output layer -> 1, 0
    model.add(keras.layers.Dense(2, activation='softmax'))

    # Defining the parameters of the model
    model.compile(optimizer='adam',
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    # Fitting the model
    model.fit(x, y, epochs=50, batch_size=32)

    return model


def testModel(model, x, y):
    """Uses the saved model and the testing data to evaluate the
    model. Returns the loss and accuracy as a List.
    """
    x = formatVariable(x)
    evaluation = model.evaluate(x, y)

    return evaluation


# Building the model
model = buildModel(ds_train, training_labels)

# Saving the model
# folder = r"C:/Users/Michael/OneDrive/Documents/GitHub/Mask/mask_model"
# model.save(folder)


# Testing the model
# ds_test = getData("testing_set")[0]  # x variable
# testing_labels = getData("testing_set")[1]  # y variable
# test = testModel(model, ds_test, testing_labels)
