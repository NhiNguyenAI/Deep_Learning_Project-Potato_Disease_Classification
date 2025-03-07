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

# ---------------------------------------------------------------------------------------------------------------------
# 1 Read data
# ---------------------------------------------------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------------------------------------------------
# 2 Create the function to train the model
# ---------------------------------------------------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------------------------------------------------
# 3 Preprocess the data
# ---------------------------------------------------------------------------------------------------------------------

# resizing the image, for the tranning model, we need to resize the image to the same size
resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1./255)
])

# ---------------------------------------------------------------------------------------------------------------------
# 4 Data Augmentation
# ---------------------------------------------------------------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
])

# ---------------------------------------------------------------------------------------------------------------------
# 5 Convolutional Neural Network
# ---------------------------------------------------------------------------------------------------------------------
n_classes = 3
CHANNELS = 3
model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    # 1st Convolutional Layer
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(BATH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
    layers.MaxPooling2D((2,2)),
    # 2nd Convolutional Layer
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    # 3rd Convolutional Layer
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    # 4th Convolutional Layer
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    # 5th Convolutional Layer
    layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    # Flatten the results to feed into a DNN
    layers.Flatten(),
    # 1st Dense Layer
    layers.Dense(64, activation='relu'),
    # 2nd Dense Layer
    layers.Dense(n_classes, activation='softmax')

])

model.build(input_shape=(BATH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

model.summary()

# ---------------------------------------------------------------------------------------------------------------------
# 6 Compile the model
# ---------------------------------------------------------------------------------------------------------------------
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# ---------------------------------------------------------------------------------------------------------------------
# 7 Train the model
# ---------------------------------------------------------------------------------------------------------------------
history = model.fit(
    dataset_train,
    epochs=50,
    batch_size = BATH_SIZE,
    verbose=1,
    validation_data=val_dataset,
)

# evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test accuracy: {test_accuracy} - Test loss: {test_loss}")

# ---------------------------------------------------------------------------------------------------------------------
# 8 Plot the accuracy and loss
# ---------------------------------------------------------------------------------------------------------------------
def plot_accuracy_loss(history):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    plt.show()

plot_accuracy_loss(history)

#---------------------------------------------------------------------------------------------------------------------
# 9 Predict the model
# First Image is Potato___Early_blight
# Result: Potato___Late_blight ---> Get misstakes
#---------------------------------------------------------------------------------------------------------------------

# Take frist image to predict
for images_batch, labels_batch in test_dataset.take(1):
    frist_image = images_batch[0].numpy().astype("uint8")
    frist_label = labels_batch[0].numpy()

    print("first image to predict")
    plt.imshow(frist_image)
    plt.show()
    print(f"class: {class_names[frist_label]} and label: {frist_label}")


# Predict the first image
for images_batch, labels_batch in test_dataset.take(1):
    batch_prediction = model.predict(images_batch)
    frist_image = images_batch[0].numpy().astype("uint8")
    frist_label = labels_batch[0].numpy()

    print(f"first image to predict, class: {class_names[frist_label]} and label: {frist_label}")
    print(f"batch_prediction: {batch_prediction[0]}")
    print(f"Predicted class: {class_names[np.argmax(batch_prediction[0])]}")
    print(f" max value: {np.argmax(batch_prediction[0])}")

