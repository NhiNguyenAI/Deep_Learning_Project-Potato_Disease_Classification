"""
************************************************************************
 *
 * make_dataset.py
 *
 * Initial Creation:
 *    Author      Nhi Nguyen
 *    Created on  2025-26-02
 *
*************************************************************************
"""
import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Read data
# --------------------------------------------------------------
directory = '../../data/raw/PlantVillage'
IMAGE_SIZE = 256
BATH_SIZE = 32
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    shuffle = True,
    image_size = (IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATH_SIZE
)

# check name of the class
class_names = dataset.class_names

# check length of the dataset
len(dataset)  # 68 -> 68*32 (bath) = 2176 images

# check one batch of the dataset
for image_batch, labels_batch in dataset.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())

# Show the first 9 images
plt.figure(figsize=(10, 10))
for images, labels in dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")



