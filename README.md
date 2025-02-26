# Deep Leanring Project: Potato Disease Classification using CNN

## Project Overview

This project is a machine learning-based classification system designed to identify various diseases affecting potatoes. Using Convolutional Neural Networks (CNN), the system can classify images of potato plants into different disease categories, assisting farmers in early detection and management of plant diseases.

The dataset consists of images of potato plants, some of which show symptoms of diseases such as Early Blight, Late Blight, and healthy potato plants. The goal is to train a deep learning model that can accurately predict the disease from an image of a potato plant.

## Prerequisites

To run this project, you will need to install the following dependencies:

- Python 8.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Pandas
- scikit-learn


## Dataset

The dataset used in this project consists of labeled images of potato plants, which can be found on various platforms like Kaggle. You can download the dataset from [this link](https://www.kaggle.com/datasets/).

The dataset is divided into several folders:
- `healthy`: Images of healthy potato plants.
- `early_blight`: Images showing Early Blight disease.
- `late_blight`: Images showing Late Blight disease.
- `other`: Other diseases affecting potato plants.


## Training the Model

To train the model, navigate to the `src` directory and run the following command: **python train_model.py**


This script will:
1. Load and preprocess the dataset.
2. Split the dataset into training and validation sets.
3. Build and compile a CNN model.
4. Train the model on the dataset.

You can adjust the hyperparameters such as learning rate, batch size, and number of epochs in the `train_model.py` script to improve performance.

## Making Predictions

Once the model is trained, you can use it to classify new images. To do this, run the following script: python predict.py --image_path /path/to/image


The model will predict the disease category of the given image and display the result.

## Evaluation

After training, evaluate the model using the validation dataset to check the accuracy and performance. You can also use metrics like confusion matrix, precision, recall, and F1-score to assess the model's performance in more detail.







