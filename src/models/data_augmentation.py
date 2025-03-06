import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import cv2

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

# --------------------------------------------------------------
# Create the dataset
# --------------------------------------------------------------
flowers_images_dict = {
    'roses': list(data_dir.glob('roses/*')),
    'daisy': list(data_dir.glob('daisy/*')),
    'dandelion': list(data_dir.glob('dandelion/*')),
    'sunflowers': list(data_dir.glob('sunflowers/*')),
    'tulips': list(data_dir.glob('tulips/*')),
}

flower_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion': 2,
    'sunflowers': 3,
    'tulips': 4,
}

# Open CV 
img = cv2.imread(str(flowers_images_dict['roses'][0]))
img.shape

# Resize the image
img = cv2.resize(img, (180, 180))
img.shape

x, y = [], []
for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        img = cv2.resize(img, (180, 180))
        x.append(img)
        y.append(flower_labels_dict[flower_name])

# --------------------------------------------------------------
# Train the model
# --------------------------------------------------------------
