import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib

# --------------------------------------------------------------
# Load the dataset
# --------------------------------------------------------------

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='flower_photos',
                                   untar=True)
data_dir = pathlib.Path(data_dir)

# --------------------------------------------------------------
# Show the dataset
# --------------------------------------------------------------

image_count = len(list(data_dir.glob('*/*.jpg')))

roses = list(data_dir.glob('roses/*'))
roses[:5]

PIL.Image.open(str(roses[1]))

tupips = list(data_dir.glob('tulips/*'))
PIL.Image.open(str(tupips[0]))

