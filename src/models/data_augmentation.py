import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Load the dataset
# --------------------------------------------------------------

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='flower_photos',
                                   untar=True)
data_dir = pathlib.Path(data_dir)

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
x, y = [], []
for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        img = cv2.resize(img, (180, 180))  # Resize images
        x.append(img)
        y.append(flower_labels_dict[flower_name])

# Convert to NumPy arrays
x = np.array(x)
y = np.array(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# Scale the data
x_train_scaled = X_train / 255.0
x_test_scaled = X_test / 255.0

# Cast to float32
x_train_scaled = x_train_scaled.astype(np.float32)
x_test_scaled = x_test_scaled.astype(np.float32)

# --------------------------------------------------------------
# Build and train the model
# --------------------------------------------------------------
num_classes = 5

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.3),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

model = tf.keras.models.Sequential([
    data_augmentation,
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(180, 180, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Use softmax for probabilities
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    x_train_scaled, y_train,
    validation_data=(x_test_scaled, y_test),
    batch_size=16,
    epochs=30
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the model
model.save('flower_classification_model.h5')

# --------------------------------------------------------------
# data augmentation
# --------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.9),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.9),
])

plt.axis('off')
plt.imshow(x[0])



plt.axis('off')
plt.imshow(data_augmentation(x)[0].numpy().astype("uint8"))



