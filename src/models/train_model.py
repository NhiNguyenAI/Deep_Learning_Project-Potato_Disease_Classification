"""
************************************************************************
 *
 * train_model.py
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

# --------------------------------------------------------------
# Create the function to train the model
# --------------------------------------------------------------

# define the parameters for the training
train_size = 0.8
val_size = 0.1

# 80% for training
dataset_train = dataset.take(int(len(dataset)*train_size))

# 10 % for testing and 10% for validation
test_dataset = dataset.skip(int(len(dataset)*train_size))
val_dataset = dataset.take(int(len(dataset)*val_size))
test_dataset = test_dataset.skip(int(len(dataset)*val_size))

# create the function get dataset partition
def get_dataset_partition(dataset, train_size, val_size, shuffe = True, shuffle_size = 10000):
    # The shuffle method is used to randomly reorder the elements in the dataset. The buffer_size=10 argument specifies the size of the buffer used for shuffling
    if shuffe:
        dataset = dataset.shuffle(shuffle_size)

    dataset_train = dataset.take(int(len(dataset)*train_size))
    test_dataset = dataset.skip(int(len(dataset)*train_size))
    val_dataset = dataset.take(int(len(dataset)*val_size))
    test_dataset = test_dataset.skip(int(len(dataset)*val_size))

    return dataset_train, test_dataset, val_dataset

dataset_train, test_dataset, val_dataset = get_dataset_partition(dataset, train_size, val_size)

# Use Cache and prefetch to optimize the performance
dataset_train = dataset_train.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
test_dataset = test_dataset.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
val_dataset = val_dataset.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)

# --------------------------------------------------------------
# Preprocess the data
# --------------------------------------------------------------

# resizing the image, for the tranning model, we need to resize the image to the same size
resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1./255)
])

# --------------------------------------------------------------
# Data Augmentation
# --------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
])

