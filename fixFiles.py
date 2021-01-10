import os
from os import listdir
from PIL import Image

# Function to rename multiple files
folder = r"C:/Users/Michael/OneDrive/Documents/GitHub/Mask/testing_set"


def renameFiles(folder):
    """Renames the files. Need to specify within the function,
    not on the call. Maybe something to add in future.
    """
    for count, filename in enumerate(os.listdir(folder)):
        dst = "correct.jpg"
        src = folder + filename
        dst = folder + dst

        # rename() function will
        # rename all the files
        os.rename(src, dst)


def resizeImages(folder, x, y):
    """Resizes an image to specified dimensions."""

    for filename in listdir(folder):
        image = Image.open(folder + '/' + filename)
        new_image = image.resize((x, y))
        new_image.save(folder + '/' + filename)


# resizeImages(folder, 400, 400)


def deleteFiles(folder, fileType):
    """Deletes all files of type fileType within the
    specified folder.
    """
    test = os.listdir(folder)

    for item in test:
        if item.endswith(fileType):
            os.remove(os.path.join(folder, item))


# deleteFiles(folder, ".jpg")


def greyscale(folder):
    """Greyscales the image to decrease the memory usage."""
    for filename in listdir(folder):
        image = Image.open(folder + '/' + filename)
        new_image = image.convert('L')
        new_image.save(folder + '/' + filename)


greyscale(folder)
