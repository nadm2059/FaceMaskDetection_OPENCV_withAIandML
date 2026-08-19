# src/detect_from_image.py
# This Python script loads a trained TensorFlow mask-detection model,
# reads an image from a file path provided through the command line,
# preprocesses the image, makes a prediction, and prints the result.


# Import OpenCV.
# OpenCV is used here to read and resize the image.
import cv2


# Import TensorFlow.
# TensorFlow is used to load the trained neural-network model
# and make predictions.
import tensorflow as tf


# Import NumPy.
# NumPy is used to work with the image as a numerical array
# and to add the batch dimension required by the model.
import numpy as np


# Import the sys module.
# sys allows the program to access command-line arguments.
import sys


# Load the trained TensorFlow/Keras model.
#
# 'models/mask_detector.h5' is the location of the saved model.
#
# The .h5 file contains the trained neural-network model,
# including its architecture and learned weights.
model = tf.keras.models.load_model(
    'models/mask_detector.h5'
)


# Get the image path from the command-line arguments.
#
# sys.argv is a list containing arguments passed to the Python script.
#
# sys.argv[0] = name of the Python script
# sys.argv[1] = first argument after the script name
#
# For example, if we run:
#
# python src/detect_from_image.py test.jpg
#
# then:
#
# sys.argv[0] = "src/detect_from_image.py"
# sys.argv[1] = "test.jpg"
img_path = sys.argv[1]


# Read the image from the file path using OpenCV.
#
# cv2.imread() loads the image into a NumPy array.
#
# The image is stored as numerical pixel values.
img = cv2.imread(img_path)


# Resize the image to 224 x 224 pixels.
#
# The trained model expects images with this size.
#
# The division by 255.0 normalizes the pixel values.
#
# Original pixel values:
#
#     0 - 255
#
# After dividing by 255:
#
#     0.0 - 1.0
#
# This normalization makes the image suitable for the neural network.
resized = cv2.resize(img, (224, 224)) / 255.0


# Add a batch dimension to the image.
#
# Before np.expand_dims():
#
#     (224, 224, 3)
#
# After np.expand_dims():
#
#     (1, 224, 224, 3)
#
# The first dimension represents the number of images being processed.
#
# Here, we are processing exactly one image.
resized = np.expand_dims(resized, axis=0)


# Pass the processed image through the neural-network model.
#
# model.predict() performs inference using the trained model.
#
# The result is stored in pred.
#
# [0] selects the first image's prediction.
#
# [0] then selects the first/output value.
#
# For example, if the model returns:
#
#     [[0.23]]
#
# then:
#
#     model.predict(resized)     -> [[0.23]]
#     model.predict(resized)[0]  -> [0.23]
#     model.predict(resized)[0][0] -> 0.23
#
# Therefore, pred becomes:
#
#     0.23
pred = model.predict(resized)[0][0]


# Determine the predicted class using a threshold of 0.5.
#
# If pred is less than 0.5:
#
#     "Mask"
#
# Otherwise:
#
#     "No Mask"
#
# This is called a conditional expression or ternary expression.
#
# It has the general form:
#
#     value_if_true if condition else value_if_false
label = "Mask" if pred < 0.5 else "No Mask"


# Print the prediction to the terminal.
#
# The f before the string creates an f-string.
#
# {label} is replaced with the value stored in label.
#
# For example:
#
#     Prediction: Mask
#
# or:
#
#     Prediction: No Mask
print(f"Prediction: {label}")
