import numpy as np
from tensorflow import keras
import pandas
from matplotlib import image, pyplot

"""Displaying the image with Pillow"""
# import PIL
# from PIL import Image
# print('Pillow Version:', PIL.__version__)
# image = Image.open('opera_house.jpg')

"""Displaying the image with matplotlib"""
data = image.imread('opera_house.jpg')
pyplot.imshow(data)
pyplot.show()
# summarize some details about the image
# show the image
# image.show()
# learningRate = 1
# bias = 1
# dataset = loadtxt('spima-indians-diabetes.csv', delimiter=',')

# x = dataset[:, 0:8]
# y = dataset[:, 8]

# model = keras.models.Sequential()
# model.add(keras.layers.Dense(12, input_dim=8, activation='relu',))
# model.add(keras.layers.Dense(8, activation='relu'))
# model.add(keras.layers.Dense(1, activation='sigmoid'))

# model.compile(loss="binary_crossentropy",
#               optimizer="adam", metrics=["accuracy"])
# model.fit(x, y, epochs=150, batch_size=10)

# _, accuracy = model.evaluate(x, y)
# print('Accuracy: %.2f' % (accuracy*100))

"""Original, unformatted code for main.py"""

# Loading in the training and testing data (x and y var for each)
# ds_train = trainingData("training_set")  # x variable
# ds_test = trainingData("testing_set")
# training_labels = trainingLabels("training_set")  # y variable
# testing_labels = trainingLabels("testing_set")

# Converting to float values from integers
# ds_train = ds_train.astype('float32').reshape((1004, 400, 400, 1))
# ds_test = ds_test.astype('float32').reshape((358, 400, 400, 1))
# print(ds_train.shape, ds_test.shape)

# Scaling values to be between 0 and 1 for performance
# ds_train /= 255
# ds_test /= 255
