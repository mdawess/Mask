from os import listdir
from matplotlib import image, pyplot
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


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
    Assist from:
    https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    """
    labels = []
    i = 0
    for filename in listdir(folder):
        # Gets each filename from folder and removes the # at the end
        parsed = ''.join([i for i in filename if not i.isdigit()])
        # Appends the filename to the list excluding .jpg
        labels.append(parsed[:-4])
        i += 1

    labels = np.array(labels)
    # Convert the labels into a one hot vector (0 and 1s)
    # Done to add a dimension and bc categorical data isn't usable
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    # Converter to get the original value
    # inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0])])
    return onehot_encoded


# x = trainingData("training_set")

# print(x.shape[0])
# print(type(trainingData('testing_set')[0]))
