# src/train.py
# This Python script trains a TensorFlow/Keras image-classification model
# to distinguish between:
#
#     with_mask
#     without_mask
#
# It uses MobileNetV2 as the base neural network and then adds
# a custom binary classification layer.


# ---------------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------------


# Import TensorFlow.
# TensorFlow is the deep-learning framework used to build,
# compile, train, and save the neural-network model.
import tensorflow as tf


# Import ImageDataGenerator.
#
# ImageDataGenerator creates batches of images for training.
# It can also preprocess and augment images.
#
# In this program, it is used for:
# - Rescaling pixel values
# - Zooming images
# - Flipping images horizontally
# - Creating a validation split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Import MobileNetV2.
#
# MobileNetV2 is a pretrained convolutional neural network
# originally trained on the ImageNet dataset.
#
# We will use it as the base/feature-extraction portion
# of our mask-detection model.
from tensorflow.keras.applications import MobileNetV2


# Import Keras layers and models.
#
# "layers" gives us neural-network layers such as:
# - Dense
# - GlobalAveragePooling2D
#
# "models" lets us create models such as Sequential.
from tensorflow.keras import layers, models


# Import Python's built-in os module.
#
# os is used later to create the "models" directory
# if it does not already exist.
import os


# ---------------------------------------------------------
# TRAINING SETTINGS
# ---------------------------------------------------------


# Set the width and height of every image.
#
# Every image will be resized to:
#
#     224 × 224
#
# MobileNetV2 commonly uses 224 × 224 input images.
IMG_SIZE = 224


# Set the number of images processed at one time.
#
# A batch size of 32 means:
#
#     The model processes 32 images,
#     calculates the error,
#     and updates its weights.
#
# Then it processes the next 32 images.
BATCH_SIZE = 32


# Specify where the training images are stored.
#
# The expected directory structure is:
#
# data/
# └── train/
#     ├── with_mask/
#     │   ├── image1.jpg
#     │   ├── image2.jpg
#     │   └── ...
#     │
#     └── without_mask/
#         ├── image1.jpg
#         ├── image2.jpg
#         └── ...
#
# flow_from_directory() automatically recognizes
# the subfolder names as class names.
train_dir = 'data/train'


# ---------------------------------------------------------
# CREATE THE DATA PIPELINE
# ---------------------------------------------------------


# Create an ImageDataGenerator.
#
# ImageDataGenerator is responsible for preparing
# the images before they are given to the neural network.
#
# It can also perform data augmentation.
datagen = ImageDataGenerator(


    # Rescale every pixel value by 1/255.
    #
    # Normal image pixels normally range from:
    #
    #     0 to 255
    #
    # Dividing by 255 converts them to approximately:
    #
    #     0.0 to 1.0
    #
    # Example:
    #
    #     255 / 255 = 1.0
    #     128 / 255 ≈ 0.502
    #       0 / 255 = 0.0
    rescale=1./255,


    # Randomly zoom into images by up to 20%.
    #
    # This is a form of data augmentation.
    #
    # It creates slightly different versions of training images,
    # which can help the model generalize to new images.
    zoom_range=0.2,


    # Randomly flip images horizontally.
    #
    # For example:
    #
    # Original:
    #
    #     Person facing left
    #
    # Flipped:
    #
    #     Person facing right
    #
    # This gives the model more variations of training images.
    horizontal_flip=True,


    # Reserve 20% of the images for validation.
    #
    # The dataset is divided into:
    #
    #     80% → training
    #     20% → validation
    #
    # Validation images are used to evaluate the model
    # during training without updating the model's weights
    # from those validation examples.
    validation_split=0.2
)


# ---------------------------------------------------------
# CREATE TRAINING DATA GENERATOR
# ---------------------------------------------------------


# Create a generator that provides the training images.
#
# flow_from_directory() automatically:
#
# 1. Looks inside train_dir.
# 2. Finds the class subfolders.
# 3. Loads the images.
# 4. Resizes the images.
# 5. Applies preprocessing/augmentation.
# 6. Assigns class labels.
# 7. Returns batches of images.
train_gen = datagen.flow_from_directory(


    # Tell Keras where the images are located.
    train_dir,


    # Resize every image to:
    #
    #     224 × 224
    target_size=(IMG_SIZE, IMG_SIZE),


    # Load 32 images at a time.
    batch_size=BATCH_SIZE,


    # Use binary classification.
    #
    # "binary" means there are two classes.
    #
    # For this project:
    #
    #     with_mask
    #     without_mask
    #
    # Keras will assign binary numerical labels
    # to these classes.
    class_mode='binary',


    # Tell Keras to use the 80% training portion
    # of the dataset.
    subset='training'
)


# ---------------------------------------------------------
# CREATE VALIDATION DATA GENERATOR
# ---------------------------------------------------------


# Create a second generator for validation images.
#
# It uses the same ImageDataGenerator,
# but selects the validation portion of the dataset.
val_gen = datagen.flow_from_directory(


    # Use the same training directory.
    train_dir,


    # Resize validation images to 224 × 224.
    target_size=(IMG_SIZE, IMG_SIZE),


    # Process validation images in batches of 32.
    batch_size=BATCH_SIZE,


    # Use binary labels:
    #
    #     0 or 1
    class_mode='binary',


    # Use the 20% validation portion of the dataset.
    subset='validation'
)


# ---------------------------------------------------------
# CREATE THE BASE MODEL
# ---------------------------------------------------------


# Create a MobileNetV2 model.
#
# input_shape=(IMG_SIZE, IMG_SIZE, 3)
#
# means the model expects:
#
#     height = 224
#     width  = 224
#     channels = 3
#
# The 3 channels represent:
#
#     Red
#     Green
#     Blue
#
#
# include_top=False
#
# removes MobileNetV2's original ImageNet classification layer.
#
# The original MobileNetV2 was designed to classify
# 1,000 ImageNet categories.
#
# We don't need those categories.
#
# We only need:
#
#     with_mask
#     without_mask
#
#
# weights='imagenet'
#
# loads weights learned from the ImageNet dataset.
#
# This is called transfer learning.
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)


# Freeze the MobileNetV2 base model.
#
# trainable=False means the pretrained MobileNetV2 weights
# will NOT be updated during our mask-detection training.
#
# Therefore, MobileNetV2 acts as a feature extractor.
base_model.trainable = False


# ---------------------------------------------------------
# BUILD THE FINAL MODEL
# ---------------------------------------------------------


# Create a Sequential model.
#
# Sequential means that the layers are connected
# in a simple sequence:
#
#     Layer 1
#       ↓
#     Layer 2
#       ↓
#     Layer 3
#       ↓
#     ...
model = models.Sequential([


    # Add the pretrained MobileNetV2 feature extractor.
    #
    # The image enters MobileNetV2 first.
    #
    # MobileNetV2 extracts useful visual features such as:
    #
    # - Edges
    # - Shapes
    # - Textures
    # - Patterns
    # - Facial features
    base_model,


    # Convert the feature maps produced by MobileNetV2
    # into a single feature vector.
    #
    # GlobalAveragePooling2D calculates the average
    # of each feature map.
    #
    # This reduces the amount of data before the final
    # classification layer.
    layers.GlobalAveragePooling2D(),


    # Create the final classification layer.
    #
    # Dense(1) means there is one output neuron.
    #
    # activation='sigmoid' converts the output into
    # a probability between 0 and 1.
    #
    # Example:
    #
    #     0.15
    #     0.72
    #     0.93
    #
    # This is appropriate for binary classification.
    layers.Dense(
        1,
        activation='sigmoid'
    )
])


# ---------------------------------------------------------
# COMPILE THE MODEL
# ---------------------------------------------------------


# Configure the model before training.
#
# model.compile() tells Keras:
#
# - Which optimizer to use
# - Which loss function to use
# - Which metrics to calculate
model.compile(


    # Use the Adam optimization algorithm.
    #
    # The optimizer determines how the model's trainable
    # weights are updated based on the error.
    optimizer='adam',


    # Use binary cross-entropy as the loss function.
    #
    # Binary cross-entropy is commonly used when there
    # are exactly two classes.
    #
    # It compares:
    #
    #     Actual label
    #
    # with:
    #
    #     Predicted probability
    loss='binary_crossentropy',


    # Track classification accuracy during training.
    #
    # Accuracy tells us what percentage of predictions
    # were classified correctly.
    metrics=['accuracy']
)


# ---------------------------------------------------------
# TRAIN THE MODEL
# ---------------------------------------------------------


# Train the neural network.
#
# train_gen provides the training images.
#
# validation_data=val_gen provides the validation images.
#
# epochs=5 means the model will go through the training
# dataset 5 complete times.
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)


# ---------------------------------------------------------
# SAVE THE TRAINED MODEL
# ---------------------------------------------------------


# Create the "models" directory if it doesn't already exist.
#
# exist_ok=True means:
#
#     If the directory already exists,
#     don't produce an error.
#
# If it doesn't exist, create it.
os.makedirs(
    'models',
    exist_ok=True
)


# Save the trained TensorFlow/Keras model.
#
# The model will be saved as:
#
#     models/mask_detector.h5
#
# This saved model can later be loaded by:
#
#     detect_from_image.py
#     detect_from_webcam.py
#     app.py
model.save(
    'models/mask_detector.h5'
)
