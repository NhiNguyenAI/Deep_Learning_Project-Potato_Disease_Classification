# --------------------------------------------------------------
# Convolutional Neural Network
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
