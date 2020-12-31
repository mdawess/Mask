from os import listdir
from matplotlib import image, pyplot
import numpy as np


def trainingData(folder):
    """Pulls the training data (images in this case) from the specified folder and
    stores them in a list. Type can either be Train or Test.
    """
    trainingImages = []
    i = 0
    for filename in listdir(folder):
        # load image
        img_data = image.imread(folder + '/' + filename)
        # store loaded image
        trainingImages.append(img_data)
        i += 1

    return np.array(trainingImages)


def trainingLabels(folder):
    """Returns the labels for each of the images in the training dataset to be used
    as the dependent variable.
    """
    labels = []
    i = 0
    for filename in listdir(folder):
        # Gets each filename from folder and removes the # at the end
        parsed = ''.join([i for i in filename if not i.isdigit()])
        # Appends the filename to the list excluding .jpg
        labels.append(parsed[:-4])
        i += 1

    return np.array(labels)
