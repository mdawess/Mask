import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
tfds.disable_progress_bar()

builder = tfds.builder('rock_paper_scissors')
info = builder.info

ds_test = tfds.load(name="rock_paper_scissors", split="test")
test_labels = np.array([example['label'].numpy() for example in ds_test])

# print(test_labels.shape)
