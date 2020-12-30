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
