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
        img_data = image.imread(
            folder + '/' + 'mask' + str(i) + '.jpg')
        # store loaded image
        trainingImages.append(img_data)
        i += 1

    return np.array(trainingImages)
